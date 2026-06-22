import xml.etree.ElementTree as ET
import os

def parse_assy(xml_path):
    if not os.path.exists(xml_path):
        print(f"File {xml_path} not found")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Map of Object Name to Label and Placement
    objects = {}
    
    # Find Document/Object
    for obj in root.findall(".//Object"):
        obj_name = obj.get("name")
        label = ""
        placement = {}
        
        # Properties
        for prop in obj.findall("Properties/Property"):
            p_name = prop.get("name")
            if p_name == "Label":
                label = prop.find("String").get("value")
            elif p_name == "Placement":
                pl = prop.find("PropertyPlacement")
                if pl is not None:
                    placement = {
                        "px": float(pl.get("Px")),
                        "py": float(pl.get("Py")),
                        "pz": float(pl.get("Pz")),
                        "angle": float(pl.get("A")),
                        "axis": (float(pl.get("Ox")), float(pl.get("Oy")), float(pl.get("Oz")))
                    }
        
        if label:
            objects[label] = placement

    print(f"{'Label':<40} | {'Px':>8} | {'Py':>8} | {'Pz':>8} | {'Angle'}")
    print("-" * 80)
    for label in sorted(objects.keys()):
        p = objects[label]
        if p:
            print(f"{label:<40} | {p['px']:>8.1f} | {p['py']:>8.1f} | {p['pz']:>8.1f} | {p['angle']:.3f}")

if __name__ == "__main__":
    parse_assy("tmp_assy/Document.xml")
