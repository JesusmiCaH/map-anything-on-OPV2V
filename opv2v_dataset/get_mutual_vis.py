import os
from collections import defaultdict

import yaml


def find_mutual_visibility(data_path, output_file=None):
    vehicle_ids = set(
        int(d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()
    )

    mutual_visibility = defaultdict(set)
    rows = []

    for ego_id in vehicle_ids:
        ego_dir = os.path.join(data_path, str(ego_id))
        yaml_files = [f for f in os.listdir(ego_dir) if f.endswith(".yaml")]
        yaml_files = sorted(yaml_files)

        for yaml_file in yaml_files:
            yaml_path = os.path.join(ego_dir, yaml_file)
            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                    visible_vehicles = data.get("vehicles", {})

                    visible_ids = [
                        vid
                        for vid in visible_vehicles
                        if vid in vehicle_ids and vid != ego_id
                    ]
                    if visible_ids:
                        rows.append(
                            {
                                "ego_id": ego_id,
                                "frame_name": yaml_file,
                                "visible_other_ids": visible_ids,
                            }
                        )

            except Exception as e:
                print(f"[Warning] Failed to read {yaml_path}: {e}")

    with open(output_file, mode="w") as f:
        yaml.dump(rows, f, sort_keys=False)

    print(f"[INFO] YAML output written to {output_file}")

    return mutual_visibility


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find mutually visible vehicles in an OPV2V data package"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the OPV2V data package (e.g., train/2021_08_16_22_26_54/)",
    )
    args = parser.parse_args()

    find_mutual_visibility(
        args.data_path,
        output_file=os.path.join(args.data_path, "mutual_visibility.yaml"),
    )


if __name__ == "__main__":
    main()
