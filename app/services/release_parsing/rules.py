"""Release vocabulary and canonical input fields; no matching/ranking rules."""

CUT_ALIASES = {
    "theatrical": "theatricalcut", "theatricalcut": "theatricalcut",
    "directorscut": "directorscut", "directorcut": "directorscut",
    "finalcut": "finalcut", "broadcastcut": "broadcastcut",
}
VERSION_LABELS = frozenset({"extended", "unrated"})
# Subtypes are never replaced by families in canonical metadata.
SOURCE_FAMILIES = {
    "bluray": "disc_hd", "bdrip": "disc_hd", "remux": "disc_hd",
    "web_dl": "web", "webrip": "web", "hdtv": "broadcast", "dvdrip": "disc_sd",
    "sdtv": "broadcast", "dsr": "broadcast", "satrip": "broadcast",
    "cam": "telesync", "hdcam": "telesync", "ts": "telesync", "tc": "telesync",
    "hdts": "telesync", "telesync": "telesync", "telecine": "telesync",
    "scr": "screener", "screener": "screener", "dvdscr": "screener",
}
SOURCE_ALIASES = {
    "bluray": "bluray", "blu-ray": "bluray", "bd": "bluray",
    "bdrip": "bdrip", "bd-rip": "bdrip", "remux": "remux",
    "web-dl": "web_dl", "webdl": "web_dl", "web_dl": "web_dl",
    "webrip": "webrip", "web-rip": "webrip", "hdtv": "hdtv",
    "dvdrip": "dvdrip", "dvd-rip": "dvdrip",
    "sdtv": "sdtv", "sd-tv": "sdtv", "dsr": "dsr", "satrip": "satrip", "sat-rip": "satrip",
    "cam": "cam", "hdcam": "hdcam", "hd-cam": "hdcam", "ts": "ts", "tc": "tc",
    "hdts": "hdts", "hd-ts": "hdts", "telesync": "telesync", "telecine": "telecine",
    "scr": "scr", "screener": "screener", "dvdscr": "dvdscr", "dvd-scr": "dvdscr",
}
VIDEO_ALIASES = {
    "h264": "h264", "h.264": "h264", "x264": "h264", "avc": "h264",
    "h265": "h265", "h.265": "h265", "x265": "h265", "hevc": "h265",
    "xvid": "mpeg4", "divx": "mpeg4", "av1": "av1",
}
AUDIO_ALIASES = {
    "aac": "aac", "ac3": "ac3", "dd": "ac3", "eac3": "eac3",
    "ddp": "eac3", "truehd": "truehd", "dts": "dts", "dtshdma": "dts_hd_ma",
    "dts-hd-ma": "dts_hd_ma", "flac": "flac",
}
SERVICE_FAMILIES = {
    "nf": "netflix", "netflix": "netflix", "amzn": "amazon", "amazon": "amazon",
    "hmax": "max", "max": "max", "dsnp": "disney", "atvp": "apple",
    "osn": "osn", "uhd": "uhd_disc", "uhd_disc": "uhd_disc", "dvd": "dvd_disc",
    "dvd_disc": "dvd_disc", "bd": "bd_disc", "bd_disc": "bd_disc",
}
SERVICE_COMPATIBILITY = {"uhd_disc": "optical_disc", "bd_disc": "optical_disc",
                         "dvd_disc": "optical_disc"}
# Known punctuation aliases can be parsed without guessing an arbitrary group's
# boundary. Unknown group spellings require explicit structured metadata.
GROUP_SUFFIX_PATTERN = (r"-(frame[ ._-]?stor|pi[ ._-]?r8|d[ ._-]?3g|nt[ ._-]?b|rovers|demand|afg)"
                        r"(?:[ ._-](?:English|orig|HI))*$")
FIELDS = (
    "imdb_id", "imdb_verified", "title", "year", "media_type", "season", "episode",
    "source", "release_group", "cut", "version", "variant_labels", "service_disc_family",
    "resolution", "video_codec", "audio_codec", "moviehash", "video_size", "hash_verified", "scope",
)
INPUT_FIELDS = frozenset(FIELDS) | {"edition"}  # legacy fixture label, NOT a narrative cut
