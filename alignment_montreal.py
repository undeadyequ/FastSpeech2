"""
Force align speech by montreal, tacotron (Not Implemented)
    montreal:
        input: wav_dir, txt_dir
        output: align_dir

@author: LUOXUAN
@time:   2021/11/26
@Status:
    align_wavs_index_f not checked
"""
import os
from shutil import copyfile, copy2
from pathlib import Path
from utils.dsp.prom.mnt_forcealign import align_wavs_montreal

class ForceAligner:
    """
    montreal:
    """
    def __init__(self, method="montreal", method_args=None):
        self.method = method
        self.method_args = method_args
        if self.method == "montreal":
            self.mon_args = self._get_montreal_aligner_parameter()
            if method_args is not None:
                self.mon_args.update(method_args)

        elif self.method == "tacotron":
            from espnet2.bin.tts_inference import Text2Speech
            self.tac_args = self._get_tacotron_aligner_parameter()
            if method_args is not None:
                self.tac_args.update(method_args)
                self.t2s = Text2Speech(self.tac_args["config"], self.tac_args["model_f"])

    def align_wavs(self, wav_dir, text_dir, out_dir=None):
        """

        Args:
            wav_dir:
            text_dir:

        Returns:
            save forceAlign in temp + fcal_final dir
        """
        if self.method == "montreal":
            ###
            align_wavs_montreal(self.mon_args, wav_dir, text_dir, out_dir)
        elif self.method == "tacotron":
            ### building !!!
            return align_wavs_tacotron(self.tac_args, text_dir)
        else:
            print("method not supported")

    def align_wavs_index_f(self, wav_f, text_f, out_dir):
        """
        align result in process_dir by id_wav and id_text index file
        process_dir includes: wavs, txts
        Args:
            wav_f:
                id ../a.wav
            text_f:
                id ../a.txt
            out_dir:
                all_wav
                all_txt
                all_align
        Returns:
        """
        # Gather all wav and txt into single wav_dir and txt_dir
        wav_dir = os.path.join(out_dir, "all_wav")
        txt_dir = os.path.join(out_dir, "all_txt")
        align_dir = os.path.join(out_dir, "all_align")

        # create wav_dir, txt_dir
        if not os.path.isdir(align_dir):
            Path(align_dir).mkdir(parents=True)
        if not os.path.isdir(wav_dir) or not os.path.isdir(txt_dir):
            Path(wav_dir).mkdir()
            Path(txt_dir).mkdir()
        gather_all_wav_txt(wav_f, text_f, wav_dir, txt_dir)

        # Execute montreal to process_dir/fcal_final
        """
        if not os.path.isdir(align_dir):
            return self.align_wavs(wav_dir, txt_dir)
        else:
            print("fail!")
        """
        return self.align_wavs(wav_dir, txt_dir, align_dir)


    def _get_montreal_aligner_parameter(self):
        parameter = {"executor": "/home/rosen/Project/espnet/tools/force_alignment/montreal-forced-aligner/bin/mfa_align",
                     "lexicon": "/home/rosen/Project/espnet/tools/force_alignment/librispeech-lexicon.txt",
                     "lang": "english"}
        return parameter

    def _get_tacotron_aligner_parameter(self):
        parameter = {
            "config": "",
            "model_f": ""
        }
        return parameter


def gather_all_wav_txt(wav_index, txt_index, wav_dir, txt_dir):
    with open(wav_index, "r") as f:
        for line in f:
            a = line.strip().split(" ")
            id, path = line.strip().split(" ", maxsplit=1)
            copy2(path, wav_dir)
    with open(txt_index, "r") as f:
        for line in f:
            id, txt = line.strip().split(" ", maxsplit=1)
            txt_path = os.path.join(txt_dir, id + ".txt")
            Path(txt_path).touch()
            with open(txt_path, "w") as fw:
                fw.write(txt)


def align_wav_tacotron(t2s_model, text):
    with open(text, "rb") as f:
        t = f.readline(text)
    wav, _, _, _, att_ws, _, _ = t2s_model(t)  # att_ws = (x_l, y_l)
    thrd = 0.7
    nstd = 0
    align = []
    for i in range(len(att_ws)):
        std, end = 0, 0
        for j in range(nstd, len(att_ws[0])):
            if att_ws[i, j] > thrd:
                std = j
            if att_ws[i, j] < thrd and std != 0:
                end = j
                nstd = end
                align.append([std, end])
    return align


def align_wavs_tacotron(t2s_model, text_dir):
    aligns = []
    for t in os.listdir(text_dir):
        if t.endswith(".txt"):
            text = os.path.join(text_dir, t)
            aligns.append(align_wav_tacotron(t2s_model, text))
    return aligns


def copy_to_txt_wav_dir(data_dir):
    """
    :param data_dir:
    :type data_dir:
    :return:
    :rtype:
    """
    pass

if __name__ == '__main__':
    fa = ForceAligner(method="montreal")
    ## small test
    #wav_dir = "/home/rosen/Project/espnet/utils/dsp/prom/test/emotion_wav"
    #text_dir = "/home/rosen/Project/espnet/utils/dsp/prom/test/emotion_txt"
    ## big test
    #wav_dir = "/home/Data/blizzard2013_part/wav/treasure_island"
    #text_dir = "/home/Data/blizzard2013_part/txt/treasure_island"

    ## error test
    wav_dir = "/home/Data/blizzard2013_part/wav/scandal"
    text_dir = "/home/Data/blizzard2013_part/txt/scandal"
    fa.align_wavs(wav_dir, text_dir)