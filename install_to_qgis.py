"""
Installer and Linker Script for Tranchot Extractor QGIS Plugin.
Links the plugin source code into the active QGIS profile directory so that
changes are immediately live in QGIS without manual copying.

Supports:
- Windows (%APPDATA%/QGIS/QGIS3/profiles/default/python/plugins)
- Linux (~/.local/share/QGIS/QGIS3/profiles/default/python/plugins)
- macOS (~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins)

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


def get_qgis_plugins_dir() -> Path:
    """Determine default QGIS 3 plugins directory depending on operating system."""
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
        return home / "AppData" / "Roaming" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
    elif system == "Darwin":  # macOS
        return home / "Library" / "Application Support" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"
    else:  # Linux / Unix
        return home / ".local" / "share" / "QGIS" / "QGIS3" / "profiles" / "default" / "python" / "plugins"


def main():
    print("=" * 65)
    print("[*] HistMap Extractor - QGIS Plugin Installer (BCDH)")
    print("=" * 65)

    # 1. Determine source plugin directory
    repo_root = Path(__file__).resolve().parent
    plugin_src = repo_root / "tranchot_qgis_plugin"

    if not plugin_src.exists():
        # Fallback: check if script is inside root or subdirectory
        alt_src = repo_root.parent / "tranchot_qgis_plugin"
        if alt_src.exists():
            plugin_src = alt_src
        else:
            print(f"❌ Fehler: Plugin-Quellordner nicht gefunden:\n   {plugin_src}")
            return 1

    # 2. Determine target QGIS plugins directory
    qgis_plugins_dir = get_qgis_plugins_dir()
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
                print(f"[i] Altes Plugin-Verzeichnis bereinigt: {old_path.name}")
            except Exception as e:
                print(f"[!] Bereinigungshinweis ({old_path.name}): {e}")

    print(f"[*] Plugin-Quelle:     {plugin_src}")
    print(f"[*] QGIS-Plugin-Pfad:  {target_plugin_link}\n")

    # 3. Remove existing link/directory if present
    if target_plugin_link.exists() or target_plugin_link.is_symlink():
        try:
            if target_plugin_link.is_symlink():
                target_plugin_link.unlink()
                print("ℹ️  Alte symbolische Verknüpfung entfernt.")
            elif target_plugin_link.is_dir():
                try:
                    os.rmdir(target_plugin_link)
                    print("ℹ️  Alte Junction-Verknüpfung entfernt.")
                except OSError:
                    shutil.rmtree(target_plugin_link)
                    print("ℹ️  Alten Plugin-Ordner entfernt.")
        except Exception as e:
            print(f"⚠️  Warnung beim Entfernen des alten Pfads: {e}")

    # 4. Create Windows Directory Junction or Symlink
    created = False
    if platform.system() == "Windows":
        # Directory junction does not require Administrator privileges on Windows!
        cmd = f'cmd /c mklink /J "{target_plugin_link}" "{plugin_src}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
        if res.returncode == 0:
            print("✅ Directory Junction erfolgreich erstellt (Live-Verknüpfung aktiv).")
            created = True
        else:
            # Fallback to copytree if junction fails
            print(f"⚠️  Junction fehlgeschlagen: {res.stderr.strip()}")
            print("📦 Kopiere stattdessen den Ordner...")
            shutil.copytree(str(plugin_src), str(target_plugin_link))
            print("✅ Ordner erfolgreich kopiert.")
            created = True
    else:
        try:
            os.symlink(str(plugin_src), str(target_plugin_link))
            print("✅ Symbolische Verknüpfung erfolgreich erstellt.")
            created = True
        except Exception as e:
            print(f"⚠️  Symlink fehlgeschlagen ({e}), kopiere Ordner...")
            shutil.copytree(str(plugin_src), str(target_plugin_link))
            print("✅ Ordner erfolgreich kopiert.")
            created = True

    # 5. Check QGIS Python environment for OpenCV and dependencies (Windows search)
    if platform.system() == "Windows":
        qgis_candidates = [
            r"C:\Program Files\QGIS 3.42.1\bin\python-qgis.bat",
            r"C:\Program Files\QGIS 3.38.1\bin\python-qgis.bat",
            r"C:\Program Files\QGIS 3.34.1\bin\python-qgis.bat",
            r"C:\OSGeo4W\bin\python-qgis.bat",
        ]
        qgis_python_bat = next((p for p in qgis_candidates if os.path.exists(p)), None)

        if qgis_python_bat:
            print(f"\n🔍 Prüfe QGIS Python-Umgebung ({qgis_python_bat})...")
            check_cmd = [qgis_python_bat, "-c", "import cv2; print('OK')"]
            cv_check = subprocess.run(check_cmd, capture_output=True, text=True, errors="replace")
            if "OK" in cv_check.stdout:
                print("✅ OpenCV ist bereits in QGIS verfügbar!")
            else:
                print("[i] OpenCV fehlt noch im QGIS-Python. Installiere 'opencv-python'...")
                install_cmd = [qgis_python_bat, "-m", "pip", "install", "--user", "--no-deps", "opencv-python"]
                pip_res = subprocess.run(install_cmd, capture_output=True, text=True, errors="replace")
                if pip_res.returncode == 0:
                    print("[+] Abhängigkeiten erfolgreich in QGIS-Benutzerumgebung installiert.")
                else:
                    print(f"⚠️  Pip-Installation meldete: {pip_res.stderr.strip()[:200]}")

    print("\n" + "=" * 65)
    print("🎉 INSTALLATION COMPLETE / INSTALLATION ABGESCHLOSSEN!")
    print("How to activate the plugin in QGIS / Aktivierung in QGIS:")
    print("1. Start / Restart QGIS 3.")
    print("2. Open menu: 'Plugins' -> 'Manage and Install Plugins...' ('Erweiterungen verwalten...')")
    print("3. Select 'Installed' ('Installiert') and check 'HistMap Extractor'.")
    print("4. Click the new toolbar icon '🗺️ HistMap Extractor'!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
