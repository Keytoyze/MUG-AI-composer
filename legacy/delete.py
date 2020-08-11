import os, json
import traceback

for root, dirs, files in os.walk(os.path.join("out", "songs")):#"G:\\E\\osu!\\Songs"):
    for f in files:
        if f == "data.json":
            js_name = os.path.join(root, f)
            try:
                js_data = json.load(open(js_name))
                delete = False
                out = {}
                for key, data in js_data.items():
                    if data['old_star'] == -1 or data['new_star'] != -1:
                        out[key] = data
                    else:
                        delete = True
                        print("delete key: " + str(data))
                if delete:
                    json.dump(out, open(js_name, "w"))
            except:
                traceback.print_exc()
                print("remove: " + js_name)
                os.remove(js_name)
