#!/usr/bin/env bash
# WSL sandbox for the High websites benchmark (owner decision 2026-10-28).
# Run ONCE as root from Windows PowerShell:
#   wsl -d Ubuntu -u root bash <home>/<local-folder>
#   wsl --shutdown            (applies /etc/wsl.conf; after this /mnt/c no longer exists inside Ubuntu)
# Then check:  wsl -d Ubuntu -u root bash /srv/hw/verify_sandbox.sh
#
# What it does
#  1. /etc/wsl.conf: no Windows drives inside Linux (automount off) and no Windows programs from Linux (interop off).
#     Backup: /etc/wsl.conf.bak-<time>. The Windows side still reaches Linux files via \\wsl.localhost\Ubuntu.
#  2. python3-flask + python3-venv from Ubuntu packages (no pip downloads during the contest).
#  3. One locked Linux user per contest entry (hw-<entry>), home /srv/hw/sites/<entry>, mode 700: an entry's code
#     cannot read another entry's site or /srv/hw/verifier (mode 700, root). Login shell: nologin.
#  4. Per-user limits (processes, open files) so a runaway server cannot take the machine down.
# The runner (on Windows) copies files in/out as root and runs model code with
#   runuser -u hw-<entry> -- prlimit --nproc=256:256 --nofile=4096:4096 -- <command>
# (runuser skips pam_limits, so limits.d alone does not apply - checked 2026-10-28),
# then `pkill -u hw-<entry>` after every cell.
set -euo pipefail

ENTRIES=(gptoss--high gptoss--medium gptoss--low gptoss--high-syshint
         qwen-coder--vendor qwen-coder--vendor-syshint qwen-coder--t03
         nemotron--vendor-th nemotron--vendor-nt nemotron--vendor-th-syshint
         glm--agent-th glm--default-th glm--agent-nt glm--agent-th-syshint
         qwen9--th-coding qwen9--th-general qwen9--th-coding-syshint qwen9--nt-general
         k2--q5-template k2--q5-template-syshint)

[ "$(id -u)" = 0 ] || { echo "run as root: wsl -d Ubuntu -u root bash $0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1. /etc/wsl.conf"
cp -a /etc/wsl.conf "/etc/wsl.conf.bak-$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
python3 - <<'PY'
import configparser
p = '/etc/wsl.conf'
c = configparser.ConfigParser()
c.optionxform = str
c.read(p)
for section, values in {'automount': {'enabled': 'false', 'mountFsTab': 'false'},
                        'interop': {'enabled': 'false', 'appendWindowsPath': 'false'}}.items():
    if not c.has_section(section):
        c.add_section(section)
    for k, v in values.items():
        c.set(section, k, v)
with open(p, 'w') as f:
    c.write(f)
print(open(p).read())
PY

echo "== 2. packages"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-flask python3-venv >/dev/null
python3 -c "import flask, sqlite3; print('flask', flask.__version__, 'sqlite', sqlite3.sqlite_version)"

echo "== 3. users and folders"
install -d -m 755 -o root -g root /srv/hw /srv/hw/sites
install -d -m 700 -o root -g root /srv/hw/verifier
for e in "${ENTRIES[@]}"; do
  u="hw-$e"
  id "$u" >/dev/null 2>&1 || useradd --system --home-dir "/srv/hw/sites/$e" --shell /usr/sbin/nologin "$u"
  install -d -m 700 -o "$u" -g "$u" "/srv/hw/sites/$e"
done
printf '%s\n' "${ENTRIES[@]}" > /srv/hw/entries.txt

echo "== 4. limits"
cat > /etc/security/limits.d/hw-sites.conf <<'EOF'
# High websites benchmark: per contest user
@hw-sites  hard  nproc   256
@hw-sites  hard  nofile  4096
EOF
groupadd -f hw-sites
for e in "${ENTRIES[@]}"; do usermod -aG hw-sites "hw-$e"; done

install -m 700 -o root -g root "$HERE/verify_sandbox.sh" /srv/hw/verify_sandbox.sh
echo
echo "Done. Now in PowerShell:  wsl --shutdown   then   wsl -d Ubuntu -u root bash /srv/hw/verify_sandbox.sh"
