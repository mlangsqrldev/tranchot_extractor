"""
Installer and Linker Script for Tranchot Extractor QGIS Plugin.
Links the plugin source code into the active QGIS profile directory so that
changes are immediately live in QGIS without manual copying.

Supports:
- Windows (%APPDATA%/QGIS/QGIS4/profiles/default/python/plugins)
- Linux (~/.local/share/QGIS/QGIS4/profiles/default/python/plugins)
- macOS (~/Library/Application Support/QGIS/QGIS4/profiles/default/python/plugins)

Developed by the Bonn Center for Digital Humanities (BCDH), University of Bonn.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_qgis_plugins_dirs() -> list:
    """Determine all QGIS 3/4 plugins directories depending on operating system and profiles."""
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        qgis_base = Path(appdata) / "QGIS" if appdata else home / "AppData" / "Roaming" / "QGIS"
    elif system == "Darwin":  # macOS
        qgis_base = home / "Library" / "Application Support" / "QGIS"
    else:  # Linux / Unix
        qgis_base = home / ".local" / "share" / "QGIS"

    target_dirs = []
    for q_dir in ["QGIS4", "QGIS3"]:
        base_dir = qgis_base / q_dir
        if base_dir.exists():
            profiles_dir = base_dir / "profiles"
            if profiles_dir.exists():
                for p in profiles_dir.iterdir():
                    if p.is_dir():
                        target_dirs.append(p / "python" / "plugins")
            else:
                target_dirs.append(base_dir / "profiles" / "default" / "python" / "plugins")

    if not target_dirs:
        target_dirs.append(qgis_base / "QGIS3" / "profiles" / "default" / "python" / "plugins")
    return target_dirs


def link_plugin(plugin_src: Path, qgis_plugins_dir: Path) -> bool:
    """Link or copy plugin into target directory."""
    qgis_plugins_dir.mkdir(parents=True, exist_ok=True)
    target_plugin_link = qgis_plugins_dir / "tranchot_qgis"
    old_plugin_link = qgis_plugins_dir / "tranchot_extractor"

    # Clean up old tranchot_extractor plugin link if present
    for old_path in [old_plugin_link, qgis_plugins_dir / "tranchot_qgis_plugin"]:
        if old_path.exists() or old_path.is_symlink():
            try:
                if old_path.is_symlink():
                    old_path.unlink()
                elif old_path.is_dir():
                    try:
                        os.rmdir(old_path)
                    except OSError:
                        shutil.rmtree(old_path)
                print(f"  [i] Altes Verzeichnis bereinigt: {old_path.name}")
            except Exception as e:
                print(f"  [!] Bereinigungshinweis ({old_path.name}): {e}")

    # Remove existing link/directory if present
    if target_plugin_link.exists() or target_plugin_link.is_symlink():
        try:
            if target_plugin_link.is_symlink():
                target_plugin_link.unlink()
            elif target_plugin_link.is_dir():
                try:
                    os.rmdir(target_plugin_link)
                except OSError:
                    shutil.rmtree(target_plugin_link)
        except Exception as e:
            print(f"  ⚠️  Warnung beim Entfernen des alten Pfads: {e}")

    # Create Windows Directory Junction or Symlink
    if platform.system() == "Windows":
        cmd = f'cmd /c mklink /J "{target_plugin_link}" "{plugin_src}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
        if res.returncode == 0:
            print(f"  ✅ Directory Junction aktiv -> {target_plugin_link}")
            return True
        else:
            print(f"  ⚠️  Junction fehlgeschlagen, kopiere Ordner...")
            shutil.copytree(str(plugin_src), str(target_plugin_link))
            print(f"  ✅ Ordner kopiert -> {target_plugin_link}")
            return True
    else:
        try:
            os.symlink(str(plugin_src), str(target_plugin_link))
            print(f"  ✅ Symlink aktiv -> {target_plugin_link}")
            return True
        except Exception as e:
            shutil.copytree(str(plugin_src), str(target_plugin_link))
            print(f"  ✅ Ordner kopiert -> {target_plugin_link}")
            return True


def main():
    print("=" * 65)
    print("[*] HistMap Extractor - QGIS Plugin Installer (BCDH)")
    print("=" * 65)

    # 1. Determine source plugin directory
    repo_root = Path(__file__).resolve().parent
    plugin_src = repo_root / "tranchot_qgis_plugin"

    if not plugin_src.exists():
        alt_src = repo_root.parent / "tranchot_qgis_plugin"
        if alt_src.exists():
            plugin_src = alt_src
        else:
            print(f"❌ Fehler: Plugin-Quellordner nicht gefunden:\n   {plugin_src}")
            return 1

    print(f"[*] Plugin-Quelle: {plugin_src}\n")

    # 2. Determine target QGIS plugins directories
    target_dirs = get_qgis_plugins_dirs()
    for t_dir in target_dirs:
        print(f"[*] Verknüpfe mit Profil: {t_dir}")
        link_plugin(plugin_src, t_dir)

    # 3. Check all QGIS Python environments for package installation
    if platform.system() == "Windows":
        import glob
        qgis_candidates = sorted(
            glob.glob(r"C:\Program Files\QGIS 4.*\bin\python-qgis.bat") +
            glob.glob(r"C:\Program Files\QGIS 3.*\bin\python-qgis.bat") +
            [r"C:\OSGeo4W\bin\python-qgis.bat"],
            reverse=True
        )

        seen_bats = set()
        for qgis_python_bat in qgis_candidates:
            if os.path.exists(qgis_python_bat) and qgis_python_bat not in seen_bats:
                seen_bats.add(qgis_python_bat)
                print(f"\n🔍 Prüfe QGIS Python-Umgebung ({qgis_python_bat})...")
                setup_path = repo_root / "setup.py"
                if setup_path.exists():
                    install_cmd = [qgis_python_bat, "-m", "pip", "install", "--user", "--no-deps", "-e", str(repo_root)]
                    pip_res = subprocess.run(install_cmd, capture_output=True, text=True, errors="replace")
                    if pip_res.returncode == 0:
                        print("  [+] tranchot_extractor erfolgreich im Entwicklungsmodus registriert.")
                    else:
                        print(f"  ⚠️  Pip-Hinweis: {pip_res.stderr.strip()[:300]}")

    print("\n" + "=" * 65)
    print("🎉 INSTALLATION COMPLETE / INSTALLATION ABGESCHLOSSEN!")
    print("How to activate the plugin in QGIS / Aktivierung in QGIS:")
    print("1. Start / Restart QGIS.")
    print("2. Open menu: 'Plugins' -> 'Manage and Install Plugins...' ('Erweiterungen verwalten...')")
    print("3. Select 'Installed' ('Installiert') and check 'HistMap Extractor'.")
    print("4. Click the new toolbar icon '🗺️ HistMap Extractor'!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
