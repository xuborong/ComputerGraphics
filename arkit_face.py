import json
data_path = "C:/Users/xubor/OneDrive/Desktop/prediction.csv"
def load_apple_motion(motion_data_path = "", recordingfps=30, calibration_data_path=""):
    name_to_mesh = {"eyesquintright": "Mesh31", "eyelookupright": "Mesh33", "jawright": "Mesh25", "mouthstretchright": "Mesh4", "browdownleft": "Mesh50", "cheekpuff": "Mesh45", "mouthsmileleft": "Mesh5", "jawopen": "Mesh26", "mouthshrugupper": "Mesh8", "mouthleft": "Mesh18", "mouthsmileright": "Mesh7", "jawforward": "Mesh28", "mouthrollupper": "Mesh10", "mouthlowerdownright": "Mesh16", "browouterupright": "Mesh46", "mouthdimpleright": "Mesh22", "eyelookdownleft": "Mesh40", "eyewideleft": "Mesh30", "mouthshruglower": "Mesh9", "mouthpressright": "Mesh14", "mouthrolllower": "Mesh11", "jawleft": "Mesh27", "cheeksquintright": "Mesh43", "mouthpucker": "Mesh13", "mouthdimpleleft": "Mesh23", "mouthupperupleft": "Mesh2", "eyelookdownright": "Mesh39", "nosesneerright": "Mesh", "eyelookoutleft": "Mesh36", "eyewideright": "Mesh29", "browinnerup": "Mesh48", "cheeksquintleft": "Mesh44", "eyelookupleft": "Mesh34", "mouthlowerdownleft": "Mesh17", "browouterupleft": "Mesh47", "eyelookinleft": "Mesh38", "eyelookinright": "Mesh37", "mouthright": "Mesh12", "mouthstretchleft": "Mesh6", "mouthpressleft": "Mesh15", "nosesneerleft": "Mesh1", "mouthupperupright": "Mesh3", "mouthclose": "Mesh24", "eyelookoutright": "Mesh35", "browdownright": "Mesh49", "eyeblinkleft": "Mesh42", "mouthfrownleft": "Mesh21", "mouthfrownright": "Mesh20", "eyeblinkright": "Mesh41", "mouthfunnel": "Mesh19", "eyesquintleft": "Mesh32"}
    mesh_to_name = {"Mesh45": "cheekpuff", "Mesh44": "cheeksquintleft", "Mesh47": "browouterupleft", "Mesh46": "browouterupright", "Mesh41": "eyeblinkright", "Mesh40": "eyelookdownleft", "Mesh43": "cheeksquintright", "Mesh42": "eyeblinkleft", "Mesh49": "browdownright", "Mesh48": "browinnerup", "Mesh34": "eyelookupleft", "Mesh35": "eyelookoutright", "Mesh36": "eyelookoutleft", "Mesh": "nosesneerright", "Mesh30": "eyewideleft", "Mesh31": "eyesquintright", "Mesh37": "eyelookinright", "Mesh33": "eyelookupright", "Mesh38": "eyelookinleft", "Mesh39": "eyelookdownright", "Mesh18": "mouthleft", "Mesh19": "mouthfunnel", "Mesh32": "eyesquintleft", "Mesh50": "browdownleft", "Mesh12": "mouthright", "Mesh13": "mouthpucker", "Mesh10": "mouthrollupper", "Mesh11": "mouthrolllower", "Mesh16": "mouthlowerdownright", "Mesh17": "mouthlowerdownleft", "Mesh14": "mouthpressright", "Mesh15": "mouthpressleft", "Mesh1": "nosesneerleft", "Mesh2": "mouthupperupleft", "Mesh3": "mouthupperupright", "Mesh4": "mouthstretchright", "Mesh5": "mouthsmileleft", "Mesh6": "mouthstretchleft", "Mesh7": "mouthsmileright", "Mesh8": "mouthshrugupper", "Mesh9": "mouthshruglower", "Mesh23": "mouthdimpleleft", "Mesh22": "mouthdimpleright", "Mesh21": "mouthfrownleft", "Mesh20": "mouthfrownright", "Mesh27": "jawleft", "Mesh26": "jawopen", "Mesh25": "jawright", "Mesh24": "mouthclose", "Mesh29": "eyewideright", "Mesh28": "jawforward"}
    # load model
    # load data from file
    columns = []
    raw_cal_data = []
    cal_data = []
    times = []
    with open(calibration_data_path) as f:
        labels = f.readline()
        columns = labels.split(",")
        columns = columns[2:]
        raw_cal_data = f.readlines()
    for i in range(0, len(raw_cal_data)):
        frame_time = raw_cal_data[i].split(",")[0]
        frame_time_list = frame_time.split(":")
        frame_hour, frame_minute, frame_second, frame_frame = frame_time_list
        frame_hour = float(frame_hour)
        frame_minute = float(frame_minute)
        frame_second = float(frame_second)
        frame_frame = float(frame_frame)
        frame_time = frame_frame/recordingfps + frame_second + frame_minute * 60 + frame_hour * 3600
        frame_cal_data = raw_cal_data[i].split(",")
        if len(frame_cal_data) > 20:
            times.append(frame_time)
            cal_data.append(frame_cal_data[2:])
    start_time = times[0]
    for i in range(0, len(times)):
        times[i] = times[i] - start_time
    for j in range(0, len(columns)-10):
        name = columns[j].lower()
        try:
            weight_name = name_to_mesh[name]
        except:
            if name[-2:] == "_l":
                name = name[:-2] + "left"
            else:
                name = name[:-2] + "right"
            weight_name = name_to_mesh[name]
        cmds.cutKey("blendShape1.{}".format(weight_name), s=True)
    cmds.cutKey("Neutral:Mesh.{}".format("rotateY"))
    cmds.cutKey("Neutral:Mesh.{}".format("rotateX"))
    cmds.cutKey("Neutral:Mesh.{}".format("rotateZ"))
    # compute curve using loaded data:
    for i in range(0, len(times)):
        for j in range(0, len(columns)-10):
            name = columns[j].lower()
            try:
                weight_name = name_to_mesh[name]
            except:
                if name[-2:] == "_l":
                    name = name[:-2] + "left"
                else:
                    name = name[:-2] + "right"
                weight_name = name_to_mesh[name]
            cmds.setKeyframe("blendShape1.{}".format(weight_name), v=float(cal_data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateY"), v=(float(cal_data[i][-9]))*-90,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateX"), v=(float(cal_data[i][-8]))*-70,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateZ"), v=(float(cal_data[i][-7]))*-45,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))


load_apple_motion(data_path, 30, data_path)