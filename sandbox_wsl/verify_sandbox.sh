#!/usr/bin/env bash
# Checks the High websites sandbox after `wsl --shutdown`. Read-only except a temporary probe file it removes.
#   wsl -d Ubuntu -u root bash /srv/hw/verify_sandbox.sh
set -uo pipefail
ok=0; bad=0
check() { if eval "$2" >/dev/null 2>&1; then echo "OK   $1"; ok=$((ok+1)); else echo "FAIL $1"; bad=$((bad+1)); fi; }

mapfile -t E < /srv/hw/entries.txt
A="hw-${E[0]}"; B="${E[1]}"
echo "probe users: $A, neighbour ${B}"
echo secret > "/srv/hw/sites/$B/probe.txt"; chown "hw-$B:hw-$B" "/srv/hw/sites/$B/probe.txt"; chmod 600 "/srv/hw/sites/$B/probe.txt"
echo secret > /srv/hw/verifier/probe.txt

check "Windows drive C: is not mounted"            "[ ! -e /mnt/c/Windows ]"
check "Windows programs cannot be started"        "! runuser -u $A -- cmd.exe /c ver"
check "entry cannot read a neighbour's site"      "! runuser -u $A -- cat /srv/hw/sites/$B/probe.txt"
check "entry cannot list a neighbour's site"      "! runuser -u $A -- ls /srv/hw/sites/$B"
check "entry cannot read the verifier"            "! runuser -u $A -- cat /srv/hw/verifier/probe.txt"
check "entry can write its own folder"            "runuser -u $A -- touch /srv/hw/sites/${E[0]}/w && rm -f /srv/hw/sites/${E[0]}/w"
check "entry cannot write outside its folder"     "! runuser -u $A -- touch /srv/hw/escape"
check "Flask and SQLite available to the entry"   "runuser -u $A -- python3 -c 'import flask, sqlite3'"
check "entry can serve on localhost"              "runuser -u $A -- python3 -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\", 0)); s.listen()'"
# runuser does not go through pam_limits, so limits.d does not apply; the runner starts model code through
# RUN (below), which sets the limits itself with prlimit - this checks that exact wrapper
RUN="runuser -u $A -- prlimit --nproc=256:256 --nofile=4096:4096 --"
check "process limit applied by the runner wrapper" "[ \"\$($RUN bash -c 'ulimit -u')\" -le 256 ] && [ \"\$($RUN bash -c 'ulimit -n')\" -le 4096 ]"

rm -f "/srv/hw/sites/$B/probe.txt" /srv/hw/verifier/probe.txt
echo; echo "passed $ok, failed $bad"
[ "$bad" = 0 ]
