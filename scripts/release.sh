#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
choice=${1:-}
if test -z "$choice"; then
    printf '%s\n' \
        'Choose release build:' \
        '  1) Linux GPU + Docker' \
        '  2) Linux GPU native' \
        '  3) Windows GPU native'
    read -r -p '> ' choice
fi

case "$choice" in
    1|docker|--docker)
        exec "$root/release/linux.sh" --docker
        ;;
    2|native|--native)
        exec "$root/release/linux.sh" --native
        ;;
    3|windows|--windows)
        case "$(uname -s)" in
            MINGW*|MSYS*|CYGWIN*) exec cmd.exe /c "${root//\//\\}\\release\\win.bat" ;;
            *)
                printf '%s\n' 'Windows GPU native must run on Windows:' \
                    '  scripts\release\win.bat' >&2
                exit 2
                ;;
        esac
        ;;
    *)
        printf 'Unknown release variant: %s\n' "$choice" >&2
        exit 2
        ;;
esac
