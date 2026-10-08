"""Credential presentation executes the actual UI JavaScript, with no upstream calls."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.utils.config_parser import encode_user_config, parse_user_config

TEMPLATE = Path("app/templates/configure.html")
PROVIDERS = (("subdl", "SubDL"), ("subsource", "SubSource"), ("opensubtitles", "OpenSubtitles"))


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = {}
        self.reveals = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids[attributes["id"]] = attributes
        if "data-reveal" in attributes:
            self.reveals[attributes["data-reveal"]] = (tag, attributes)


def run_js(code):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute configuration UI JavaScript")
    result = subprocess.run([node, "-e", code], capture_output=True, text=True,
                            check=True, timeout=10)
    return json.loads(result.stdout)


@pytest.mark.parametrize("provider,label", PROVIDERS)
def test_masked_inputs_and_accessible_reveal_controls(client, provider, label):
    elements = Elements(client.get("/configure").text)
    field = elements.ids[provider + "-key"]
    assert field["type"] == "password"
    assert "credential-input" in field["class"]
    assert field["autocomplete"] == "off"
    assert field["aria-label"] == label + " API key"
    tag, button = elements.reveals[provider + "-key"]
    assert tag == "button" and button["type"] == "button"
    assert button["aria-label"] == button["title"] == f"Show {label} API key"
    assert button["aria-controls"] == provider + "-key"
    assert button["aria-pressed"] == "false"


def test_credential_prefill_and_password_privacy_unchanged(client):
    token = encode_user_config(subdl_key="demo-subdl", subsource_key="demo-subsource",
                               opensubtitles_key="demo-os", opensubtitles_username="demo-user",
                               opensubtitles_password="demo-password", enable_subdl=False)
    fields = Elements(client.get(f"/{token}/configure").text).ids
    for field, value in [("subdl-key", "demo-subdl"), ("subsource-key", "demo-subsource"),
                         ("opensubtitles-key", "demo-os"), ("opensubtitles-username", "demo-user")]:
        assert fields[field]["value"] == value
    assert fields["opensubtitles-password"]["value"] == ""


DOM_SETUP = r"""
const assert = require('node:assert/strict');
const fields = ['subdl-key','subsource-key','opensubtitles-key','opensubtitles-username','opensubtitles-password'];
const makeElement = () => ({disabled:false,checked:true,value:'demo-value',type:'password',
    attrs:{},dataset:{},events:{},classes:new Set(),
    setAttribute(k,v){this.attrs[k]=v}, removeAttribute(k){delete this.attrs[k]},
    addEventListener(k,v){this.events[k]=v},
    classList:{toggle(){}}});
const elements = Object.fromEntries(fields.map(id=>[id,makeElement()]));
const buttons=[];
const wrappers={enableSubdl:'subdlKey',enableSubsource:'subsourceKey',enableOpensubtitles:'opensubtitlesKey'};
for (const [toggleId,wrapperId] of Object.entries(wrappers)) {
    const provider=wrapperId.replace('Key','');
    const input=elements[provider+'-key'];
    const button=makeElement(); button.dataset={reveal:provider+'-key',label:provider+' API key'};
    buttons.push(button);
    const link=makeElement();
    const controls=[input,button];
    if(provider==='opensubtitles') controls.push(elements['opensubtitles-username'],elements['opensubtitles-password']);
    const wrapper=makeElement();
    wrapper.link=link; wrapper.controls=controls;
    wrapper.classList={toggle:(name,on)=>on?wrapper.classes.add(name):wrapper.classes.delete(name)};
    wrapper.querySelectorAll=s=>s==='input, button'?controls:s==='a'?[link]:[button];
    elements[wrapperId]=wrapper; elements[toggleId]=makeElement();
}
const document={getElementById:id=>elements[id]||{checked:true,value:''},querySelectorAll:()=>buttons};
"""


def test_reveal_toggle_disabled_controls_and_serialization_preserve_values():
    template = TEMPLATE.read_text(encoding="utf-8")
    controls = template[template.index("const PROVIDER_KEY_WRAPPERS"):
                        template.index("function updateBadgePreview()")]
    serializer = template[template.index("function serializeConfig()"):
                          template.index("function updateUrls()")]
    actions = r"""
setupCredentialReveal();
for(const button of buttons){
    const input=elements[button.dataset.reveal]; const value=input.value;
    button.events.click(); assert.equal(input.type,'text'); assert.equal(input.value,value);
    assert.equal(button.attrs['aria-pressed'],'true'); assert.match(button.attrs['aria-label'],/^Hide /);
    button.events.click(); assert.equal(input.type,'password'); assert.equal(input.value,value);
}
for(const [toggleId,wrapperId] of Object.entries(wrappers)){
    elements[toggleId].checked=false;
}
updateProviderCollapse();
for(const [toggleId,wrapperId] of Object.entries(wrappers)){
    const wrapper=elements[wrapperId]; assert.ok(wrapper.classes.has('credentials-disabled'));
    assert.equal(wrapper.attrs['aria-disabled'],'true'); assert.equal(wrapper.link.attrs.tabindex,'-1');
    for(const control of wrapper.controls) assert.equal(control.disabled,true);
    assert.equal(elements[toggleId].disabled,false);
    const button=wrapper.controls[1]; button.events.click(); assert.equal(wrapper.controls[0].type,'password');
}
const getSubdlInput=()=>elements['subdl-key'];
const getSubsourceInput=()=>elements['subsource-key'];
const getOpenSubtitlesInput=()=>elements['opensubtitles-key'];
const getOpenSubtitlesUsernameInput=()=>elements['opensubtitles-username'];
const getOpenSubtitlesPasswordInput=()=>elements['opensubtitles-password'];
const DEFAULT_BADGE_PARTS=['score','provider','filename','uploader'];
const getBadgeParts=()=>DEFAULT_BADGE_PARTS;
const getSelectedLanguages=()=>['ara'];
const getEffectiveBaseUrl=()=> 'http://test.invalid';
const btoa=s=>Buffer.from(s,'binary').toString('base64');
const token=getManifestUrl().split('/')[3];
for(const [toggleId,wrapperId] of Object.entries(wrappers)) elements[toggleId].checked=true;
updateProviderCollapse();
for(const wrapperId of Object.values(wrappers)){
    const wrapper=elements[wrapperId]; assert.ok(!wrapper.classes.has('credentials-disabled'));
    assert.equal(wrapper.link.attrs.tabindex,undefined);
    for(const control of wrapper.controls) assert.equal(control.disabled,false);
}
assert.ok(fields.every(id=>elements[id].value==='demo-value'));
console.log(JSON.stringify(token));
"""
    token = run_js(DOM_SETUP + controls + serializer + actions)
    prefs = parse_user_config(token)
    assert prefs.subdl_key == prefs.subsource_key == prefs.opensubtitles_key == "demo-value"
    assert prefs.opensubtitles_username == prefs.opensubtitles_password == "demo-value"
    assert not prefs.enable_subdl and not prefs.enable_subsource and not prefs.enable_opensubtitles


@pytest.mark.parametrize("data,expected", [
    ({"valid": True}, "API verified"),
    ({"valid": True, "account_checked": False}, "API verified"),
    ({"valid": True, "account_checked": True, "account_valid": True}, "API & account verified"),
    ({"valid": True, "account_checked": True, "account_valid": False}, "API verified; account login rejected"),
    ({"valid": True, "account_checked": True}, "API verified"),
    ({"valid": True, "account_checked": "true", "account_valid": "true"}, "API verified"),
    ({"valid": False, "account_checked": True, "account_valid": True}, "API not verified"),
])
def test_os_status_reflects_backend_verification_flags(data, expected):
    template = TEMPLATE.read_text(encoding="utf-8")
    source = template[template.index("function openSubtitlesVerificationLabel"):
                      template.index("async function verifyOpenSubtitlesKey")]
    assert run_js(source + "console.log(JSON.stringify(openSubtitlesVerificationLabel(" +
                  json.dumps(data) + "))); ") == expected


@pytest.mark.parametrize("changed", [False, True])
def test_actual_os_validation_renders_truthfully_and_discards_stale_credentials(changed):
    template = TEMPLATE.read_text(encoding="utf-8")
    source = template[template.index("function openSubtitlesVerificationLabel"):
                      template.index("if (opensubtitlesInput) {")]
    setup = """
const status={innerHTML:''};
const key={value:'demo-key'}, user={value:'demo-user'}, password={value:'demo-password'};
const getOpenSubtitlesStatus=()=>status;
const getOpenSubtitlesInput=()=>key;
const getOpenSubtitlesUsernameInput=()=>user;
const getOpenSubtitlesPasswordInput=()=>password;
const checkIconSvg=crossIconSvg=spinIconSvg='';
const updateUrls=()=>{};
"""
    setup += "const changed=" + json.dumps(changed) + ";"
    setup += """
const fetch=async()=>{if(changed) password.value='edited'; return {json:async()=>({valid:true,account_checked:true,account_valid:false})}};
"""
    result = run_js(setup + source + "verifyOpenSubtitlesKey(key.value).then(()=>console.log(JSON.stringify(status.innerHTML))); ")
    assert "API & account verified" not in result
    if changed:
        assert "API verified" not in result
    else:
        assert "API verified; account login rejected" in result
        assert "Connection Error" not in result


def test_dark_autofill_and_disabled_css_covers_all_credentials(client):
    html = client.get("/configure").text
    fields = Elements(html).ids
    for field in [p + "-key" for p, _ in PROVIDERS] + ["opensubtitles-username", "opensubtitles-password"]:
        assert "credential-input" in fields[field]["class"]
    for suffix in ["", ":hover", ":focus", ":active"]:
        assert ".credential-input:-webkit-autofill" + suffix in html
    assert "-webkit-text-fill-color: var(--w)" in html
    assert "1000px var(--card) inset !important" in html
    assert "border-radius: 10px !important" in html
    assert ".credential-input:-webkit-autofill:focus { border-color: var(--line-h)" in html
    assert ".provider-key.credentials-disabled" in html
    assert "opacity: .45" in html
    assert ">Account</div>" in html
    assert "Optional credentials for authenticated access." in html
    assert ">Username</label>" in html
    assert ">Password</label>" in html
    assert re.findall(r'<span class="provider-name">([^<]+)</span>', html) == [label for _, label in PROVIDERS]


def test_account_simplification_preserves_mobile_geometry():
    import subprocess

    before = subprocess.run(
        ["git", "show", "HEAD:app/templates/configure.html"],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    after = TEMPLATE.read_text(encoding="utf-8")
    # Lock all responsive rules and the outer container to the accepted UI.
    assert re.findall(r"@media[^}]+(?:}[^}]*})?", after) == re.findall(
        r"@media[^}]+(?:}[^}]*})?", before)
    container = 'mx-auto w-full max-w-2xl px-4 sm:px-6 py-8 sm:py-12 pb-16 overflow-x-hidden'
    assert container in before and container in after
    assert '.account-credentials { margin-top: 12px; padding-top: 10px;' in after
