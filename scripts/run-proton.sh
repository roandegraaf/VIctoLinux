#!/usr/bin/env bash
# Runs victory.exe on Linux under a chosen Proton (or Wine) build, with
# vkd3d-proton (D3D12) and DXVK's dxgi.dll from GE-Proton, fetched once into
# a cache outside the repository. See docs/linux-proton.md.
#
#   scripts/run-proton.sh fetch                      # only fetch the DLLs
#   scripts/run-proton.sh <game exe>                 # victory run <game exe>
#   scripts/run-proton.sh <game exe> --execute       # ... and run the title
#   scripts/run-proton.sh -- config                  # any victory command
#   scripts/run-proton.sh --proton GE-Proton11-7-x86_64 --diag <game exe> --execute
#
# As cargo's runner (its first argument is the test exe under target/):
#   CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=$PWD/scripts/run-proton.sh \
#       cargo xwin test --workspace --target x86_64-pc-windows-msvc
#
# Options (each also an environment variable):
#   --proton NAME|DIR|WINE  PROTON       Proton folder, or its name under Steam's
#                                        common/ or compatibilitytools.d/, or a
#                                        wine binary (default: Proton - Experimental)
#   --game EXE              GAME_EXE     the title's exe, when not given as argument
#   --exe EXE               VICTORY_EXE  victory.exe to run (default: release build)
#   --diag                               the diagnostics build instead
#   --dlls ge|proton|DIR    VICTORY_DLLS where d3d12, d3d12core and dxgi come from:
#                                        GE-Proton's (default), the chosen Proton's
#                                        own, or a folder holding the three
#   --ge-tag TAG            GE_TAG       GE-Proton release to fetch (GE-Proton11-14)
#   --d3dcompiler b|n       D3DCOMPILER  d3dcompiler_47: Wine's builtin (default) or
#                                        a native one installed in the prefix
#   --timeout SECONDS       TIMEOUT      end the run after this long
#   --mem-max SIZE          MEM_MAX      memory cap (systemd-run MemoryMax, e.g. 20G)
#   --prefix DIR            PREFIX       Wine prefix (default: one per Proton build)
#   --auto-enter [F:T:S]    AUTO_ENTER   press Enter (the pad's A) every S seconds from
#                                        F to T seconds after input starts, to get
#                                        through the first-boot and start screens
#                                        unattended (default 15:150:4; implies --diag,
#                                        whose VICTORY_KEYS does the pressing)
#
# Also read: VICTORY_KEYS, VICTORY_LOG and the other VICTORY_* switches are
# passed through. VICTORY_NO_WATCHDOG defaults to 1 (it only has an effect
# with upstream #454 merged). Logs go to $XDG_STATE_HOME/victory/logs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/victory"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/victory"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/victory"

PROTON="${PROTON:-Proton - Experimental}"
GAME_EXE="${GAME_EXE:-}"
VICTORY_EXE="${VICTORY_EXE:-}"
VICTORY_DLLS="${VICTORY_DLLS:-ge}"
GE_TAG="${GE_TAG:-GE-Proton11-14}"
D3DCOMPILER="${D3DCOMPILER:-b}"
TIMEOUT="${TIMEOUT:-}"
MEM_MAX="${MEM_MAX:-}"
PREFIX="${PREFIX:-}"
AUTO_ENTER="${AUTO_ENTER:-}"
diag=0

die() { echo "run-proton: $*" >&2; exit 1; }
note() { echo "run-proton: $*" >&2; }

args=()
while (($#)); do
    case "$1" in
        --proton) PROTON="$2"; shift 2 ;;
        --game) GAME_EXE="$2"; shift 2 ;;
        --exe) VICTORY_EXE="$2"; shift 2 ;;
        --diag) diag=1; shift ;;
        --dlls) VICTORY_DLLS="$2"; shift 2 ;;
        --ge-tag) GE_TAG="$2"; shift 2 ;;
        --d3dcompiler) D3DCOMPILER="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --mem-max) MEM_MAX="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --auto-enter)
            if [[ "${2:-}" =~ ^[0-9]+:[0-9]+:[0-9]+$ ]]; then AUTO_ENTER="$2"; shift 2
            else AUTO_ENTER=15:150:4; shift; fi ;;
        --) shift; args+=("$@"); break ;;
        *) args+=("$1"); shift ;;
    esac
done

# --- the DLLs from GE-Proton -------------------------------------------------

ge_dir() { echo "$CACHE/runtime/$GE_TAG"; }

fetch_ge() {
    local dir; dir="$(ge_dir)"
    [[ -f "$dir/d3d12.dll" && -f "$dir/d3d12core.dll" && -f "$dir/dxgi.dll" ]] && return
    local base="https://github.com/GloriousEggroll/proton-ge-custom/releases/download/$GE_TAG"
    local name="$GE_TAG-x86_64.tar.gz" dl="$CACHE/downloads"
    mkdir -p "$dl" "$dir"
    note "fetching $name (about 600 MB, once)"
    curl -fL --retry 3 -o "$dl/$name.part" "$base/$name"
    curl -fsSL -o "$dl/$name.sha512sum" "$base/$GE_TAG-x86_64.sha512sum"
    mv "$dl/$name.part" "$dl/$name"
    (cd "$dl" && sha512sum -c "$name.sha512sum") || die "checksum mismatch for $name"
    local tmp; tmp="$(mktemp -d "$dl/extract.XXXXXX")"
    tar -xzf "$dl/$name" -C "$tmp" --wildcards \
        '*/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12.dll' \
        '*/files/lib/wine/vkd3d-proton/x86_64-windows/d3d12core.dll' \
        '*/files/lib/wine/dxvk/x86_64-windows/dxgi.dll'
    find "$tmp" -name '*.dll' -exec cp {} "$dir/" \;
    rm -rf "$tmp" "$dl/$name" "$dl/$name.sha512sum"
    # vkd3d-proton keeps its version as a plain string, e.g. "3.2.0".
    strings "$dir/d3d12core.dll" | grep -m1 -xE '[0-9]+\.[0-9]+(\.[0-9]+)?' \
        > "$dir/vkd3d-proton.version" || true
    note "vkd3d-proton $(cat "$dir/vkd3d-proton.version" 2>/dev/null || echo '?') and DXVK dxgi in $dir"
}

if [[ "${args[0]:-}" == fetch ]]; then
    fetch_ge
    exit 0
fi

# --- Proton or Wine ------------------------------------------------------------

proton_dir="" wine=""
for d in "$PROTON" \
         "$HOME/.steam/steam/steamapps/common/$PROTON" \
         "$HOME/.local/share/Steam/steamapps/common/$PROTON" \
         "$HOME/.steam/steam/compatibilitytools.d/$PROTON" \
         "$HOME/.local/share/Steam/compatibilitytools.d/$PROTON"; do
    if [[ -x "$d/files/bin/wine" ]]; then proton_dir="$(cd "$d" && pwd)"; wine="$d/files/bin/wine"; break; fi
done
if [[ -z "$wine" ]]; then
    if [[ -x "$PROTON" && "$(basename "$PROTON")" == wine* ]]; then wine="$PROTON"
    else wine="$(command -v "$PROTON" 2>/dev/null || true)"; fi
    [[ -n "$wine" && "$(basename "$wine")" == wine* ]] || die "no Proton or Wine called '$PROTON'"
fi
build="$(basename "${proton_dir:-$wine}")"

PREFIX="${PREFIX:-$DATA/prefixes/${build// /_}}"
if [[ ! -f "$PREFIX/pfx/system.reg" && ! -f "$PREFIX/system.reg" ]]; then
    mkdir -p "$PREFIX"
    if [[ -n "$proton_dir" ]]; then
        note "making a $build prefix in $PREFIX"
        STEAM_COMPAT_DATA_PATH="$PREFIX" \
        STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.steam/steam}" \
        SteamAppId=0 SteamGameId=0 WINEDEBUG=-all \
            python3 "$proton_dir/proton" run cmd /c exit >/dev/null 2>&1 || true
    else
        mkdir -p "$PREFIX/pfx"
        WINEPREFIX="$PREFIX/pfx" WINEDEBUG=-all "$wine" wineboot -u >/dev/null 2>&1 || true
    fi
fi
export WINEPREFIX="$PREFIX/pfx"
[[ -f "$WINEPREFIX/system.reg" ]] || die "could not make a prefix in $WINEPREFIX"

# --- D3D12 and DXGI ------------------------------------------------------------

case "$VICTORY_DLLS" in
    ge) fetch_ge; dlls="$(ge_dir)" ;;
    proton)
        [[ -n "$proton_dir" ]] || die "--dlls proton needs a Proton build"
        dlls="" ;;
    *) dlls="$VICTORY_DLLS" ;;
esac
if [[ -n "$dlls" ]]; then
    for f in d3d12.dll d3d12core.dll dxgi.dll; do
        [[ -f "$dlls/$f" ]] || die "$dlls has no $f"
        cmp -s "$dlls/$f" "$WINEPREFIX/drive_c/windows/system32/$f" ||
            cp "$dlls/$f" "$WINEPREFIX/drive_c/windows/system32/$f"
    done
elif [[ -n "$proton_dir" ]]; then
    # Put the Proton build's own back, in case another run replaced them.
    for f in d3d12.dll d3d12core.dll; do
        cp "$proton_dir/files/lib/wine/vkd3d-proton/x86_64-windows/$f" "$WINEPREFIX/drive_c/windows/system32/$f"
    done
    cp "$proton_dir/files/lib/wine/dxvk/x86_64-windows/dxgi.dll" "$WINEPREFIX/drive_c/windows/system32/dxgi.dll"
fi

export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n;d3dcompiler_47=$D3DCOMPILER${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export WINEDEBUG="${WINEDEBUG:--all}"
export VICTORY_NO_WATCHDOG="${VICTORY_NO_WATCHDOG:-1}"
mkdir -p "$CACHE/vkd3d" "$STATE/logs"
export VKD3D_SHADER_CACHE_PATH="${VKD3D_SHADER_CACHE_PATH:-$CACHE/vkd3d}"
export DXVK_LOG_PATH="${DXVK_LOG_PATH:-$STATE/logs}"
# Wine maps / to Z:.
winpath() { local p; p="$(realpath -m "$1")"; printf 'Z:%s' "${p//\//\\}"; }
if [[ -z "${VICTORY_CONFIG:-}" && -f "${XDG_CONFIG_HOME:-$HOME/.config}/victory/victory.toml" ]]; then
    VICTORY_CONFIG="$(winpath "${XDG_CONFIG_HOME:-$HOME/.config}/victory/victory.toml")"
    export VICTORY_CONFIG
fi

# --- what to run ---------------------------------------------------------------

# cargo's runner: a test exe of this checkout, run as is from cargo's cwd.
if [[ "${args[0]:-}" == *.exe && "$(realpath -m "${args[0]}")" == "$ROOT/target/"* ]]; then
    exec "$wine" "${args[@]}"
fi

if [[ -n "$AUTO_ENTER" ]]; then
    diag=1
    IFS=: read -r from to step <<<"$AUTO_ENTER"
    keys=""
    for ((t = from; t <= to; t += step)); do keys+="${keys:+,}$t:Enter"; done
    export VICTORY_KEYS="${VICTORY_KEYS:+$VICTORY_KEYS,}$keys"
    note "pressing Enter every ${step}s from ${from}s to ${to}s"
fi
if [[ -z "$VICTORY_EXE" ]]; then
    profile=release; ((diag)) && profile=diag
    VICTORY_EXE="$ROOT/target/x86_64-pc-windows-msvc/$profile/victory.exe"
fi
[[ -f "$VICTORY_EXE" ]] || die "no $VICTORY_EXE; build it with cargo xwin build (see docs/linux-proton.md)"

if [[ "${args[0]:-}" == *.exe && -f "${args[0]}" ]]; then
    GAME_EXE="${args[0]}"; args=("${args[@]:1}")
fi
if [[ -n "$GAME_EXE" ]]; then
    [[ -f "$GAME_EXE" ]] || die "no game exe at $GAME_EXE"
    args=(run "$(winpath "$GAME_EXE")" "${args[@]}")
fi
((${#args[@]})) || die "nothing to run: give the game's exe, or -- <victory command>"

if ! pactl info 2>/dev/null | grep -q '^Default Sink:'; then
    note "warning: no default audio sink; the title stops about 15 s in without one (Marvisak/VIctory#834)"
fi

cmd=("$wine" "$VICTORY_EXE" "${args[@]}")
[[ -n "$TIMEOUT" ]] && cmd=(timeout --signal=INT --kill-after=20 "$TIMEOUT" "${cmd[@]}")
[[ -n "$MEM_MAX" ]] && cmd=(systemd-run --user --scope --quiet -p "MemoryMax=$MEM_MAX" -p MemorySwapMax=0 "${cmd[@]}")

log="$STATE/logs/victory-$(date +%Y%m%d-%H%M%S).log"
{
    echo "# $(date -Is) $build, dlls=$VICTORY_DLLS${dlls:+ ($(cat "$dlls/vkd3d-proton.version" 2>/dev/null || echo "?"))}, d3dcompiler_47=$D3DCOMPILER"
    echo "# prefix $WINEPREFIX"
    echo "# $VICTORY_EXE ${args[*]}"
} | tee "$log" >&2
# Run from a folder outside the repository: DXVK and vkd3d write next to the cwd.
cd "$STATE"
set +e
"${cmd[@]}" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
note "exit $status; log $log"
exit "$status"
