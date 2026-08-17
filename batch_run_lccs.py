import os
import subprocess
import sys

import json
from pathlib import Path

# ===== CONFIG =====
ages = ["2000", "2006", "2011", "2015", "2020", "2033", "2040", "2048",
        "2056", "2068", "2094", "2100", "2113", "2120", "2133", "2140",
        "2154", "2165", "2180", "2200", "2210", "2220", "2230", "2240",
        "2250", "2270", "2290", "2300", "2315", "2331", "2350", "2370",
        "2383", "2393", "2408", "2420", "2444", "2463", "2475", "2489",
        "2500", "2518", "2535", "2545"]

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

project_root = _config["projet_root"]

template_path = os.path.join(project_root, "LE_DT_template.yml")
configs_folder = os.path.join(project_root, "test_processing", "configs")
lccs_script = r".\livingearth_lccs\bin\le_lccs_odc.py"  # relative, same as your manual call

# input/output roots, used here only to pre-create the per-age output folder
# (le_lccs_odc.py writes into Outputs:*:output_file but may not create the
# parent folder itself)
output_base_folder = os.path.join(project_root, "test_processing", "output")
# ==================

os.makedirs(configs_folder, exist_ok=True)

with open(template_path, "r") as f:
    template_text = f.read()

for age in ages:
    print(f"\n{'=' * 50}")
    print(f"Running LCCS classification for age: {age}")
    print(f"{'=' * 50}")

    # Make sure the per-age output folder exists before le_lccs_odc.py tries
    # to write into it
    os.makedirs(os.path.join(output_base_folder, age), exist_ok=True)

    # Render this age's YAML from the template
    rendered_yaml = template_text.format(age=age)
    config_path = os.path.join(configs_folder, f"LE_DT_{age}.yml")
    with open(config_path, "w") as f:
        f.write(rendered_yaml)

    # Run the classification for this age.
    # cwd=project_root matters here: lccs_script is a relative path
    # (".\livingearth_lccs\bin\le_lccs_odc.py"), same as your manual
    # "python .\livingearth_lccs\bin\le_lccs_odc.py .\LE_DT.yml" call, which
    # only resolves correctly when run from inside LivingEarth_DeepTime.
    # Belt-and-suspenders: explicitly put the livingearth_lccs folder (which
    # contains the le_lccs package) on PYTHONPATH for the subprocess. This
    # matters because le_lccs was installed editable (pip install -e .) into
    # a specific environment/interpreter -- if the subprocess ends up using
    # a different one, "import le_lccs" fails even though sys.executable
    # matches this script's own interpreter.
    env = os.environ.copy()
    livingearth_lccs_folder = os.path.join(project_root, "livingearth_lccs")
    env["PYTHONPATH"] = livingearth_lccs_folder + os.pathsep + env.get("PYTHONPATH", "")

    # Same reasoning for PROJ_DATA/PROJ_LIB: previously fixed per-session with
    # $env:PROJ_DATA / $env:PROJ_LIB in PowerShell, which only applies to that
    # shell. Set them here so the subprocess always points rasterio/GDAL at
    # its own bundled proj.db instead of picking up PostgreSQL/PostGIS's
    # older, incompatible one (DATABASE.LAYOUT.VERSION.MINOR mismatch).
    proj_data_path = r"C:\Users\franzisf\PycharmProjects\.venv\Lib\site-packages\rasterio\proj_data"
    env["PROJ_DATA"] = proj_data_path
    env["PROJ_LIB"] = proj_data_path

    result = subprocess.run(
        [sys.executable, lccs_script, config_path],
        capture_output=True,
        text=True,
        cwd=project_root,
        env=env
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"  ERROR (age {age}), return code {result.returncode}:")
        print(result.stderr)
    else:
        print(f"  [OK] age {age} complete")

print("\nBatch run complete.")
