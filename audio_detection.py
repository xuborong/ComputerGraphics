import maya.cmds as cmds
import time


class AudioMonitor:
    def __init__(self):
        self.known_audio_nodes = set(cmds.ls(type='audio'))

    def check_for_new_audio(self):
        current_audio_nodes = set(cmds.ls(type='audio'))
        new_nodes = current_audio_nodes - self.known_audio_nodes
        if new_nodes:
            for node in new_nodes:
                file_path = cmds.getAttr(f"{node}.filename")
                print(f"New audio file detected: {file_path}")
        self.known_audio_nodes = current_audio_nodes

        # Re-run this check after some time
        cmds.scriptJob(runOnce=True, idleEvent=self.check_for_new_audio)


audio_monitor = AudioMonitor()
audio_monitor.check_for_new_audio()
