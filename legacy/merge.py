import shutil, os, json

for root, dirs, files in os.walk(os.path.join("G:\\E\\osu!\\Songs\\")):
    for f in files:
        if f.endswith(".osu"):
            js_name = os.path.join(root, "data.json")
            mirror_js = os.path.join("out2", "out", "songs", js_name.replace("G:\\E\\osu!\\Songs\\", ""))
            if os.path.exists(mirror_js):
                js2 = json.load(open(mirror_js))
                if os.path.exists(js_name):
                    js = json.load(open(js_name))
                else:
                    js = {}

                # if len(js) != 0:
                #     print("js: " + str(js))
                #     print("js2: " + str(js2))
                js2.update(js)
                #     print("merge: " + str(js2))
                json.dump(js2, open(js_name, "w"))