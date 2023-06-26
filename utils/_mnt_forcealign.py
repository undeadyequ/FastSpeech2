"""
1. Execute montreal force aligner:
    /bin/mfa_align input_dir -c lexicon.txt english out_dir
        input_dir  : includes both a.txt and a.wav file
        lexicon.txt:
        out_dir    : the out root dir includes
                        - an folder
                            named input_dir, that includes a.TextGrid
                        - oovs_found.txt
                            includes out of ?
                        - utterance_oovs.txt
@author: LUOXUAN
@time:   2021/11/26
@Status: Checked
"""

import os
import pathlib
import subprocess
import re
from distutils.dir_util import copy_tree
import shutil
from shutil import copyfile, copy2

temp_root = "/home/rosen/temp"
temp_wavtext_dir = temp_root + "/temp_wavtext"
pathlib.Path(temp_wavtext_dir).mkdir(parents=True, exist_ok=True)

def align_wavs_montreal(mon_args, wav_dir, text_dir, output_textgird_dir=None):
    """
    Execute montreal part by part and saved to fcal_final dir

    temp_wavtext_dir:
        dir contains temp wav text pairs
    temp_textgrid_dir:
        dir contains textgrid files created from temp_wavtext_dir
    temp_textgrid_mnt_dir:

    final_textgrid_dir:
        dir saved in temp, contains all textgrid files
    output_textgird_dir:
        dir named {wav_dir}_force saved in parent of wav_dir, with contents same as final_textgrid_dir

    Args:
        mon_args:
        wav_dir:
        text_dir:
        out_temp_dir:
    Returns:
        output_textgird_dir:
            id_1.wav
            id_1.txt
            id_1.TextGrid       <- generate by forcealigner
    """

    ## when execute montreal, out_dir contents will be cleared !!
    temp_wavtext_dir = temp_root + "/temp_wavtext"    # current dir
    temp_textgrid_dir = temp_root + "/temp_wavtext_force"  # decided by montreal
    temp_textgrid_mnt_dir = temp_textgrid_dir + "/temp_wavtext"
    final_textgrid_dir = temp_root + "/force_final"
    if output_textgird_dir is None:
        output_textgird_dir = os.path.join(os.path.dirname(wav_dir), "force", os.path.basename(wav_dir))

    # Empty temp_wavtext_dir and final_textgrid_dir
    if os.path.exists(temp_wavtext_dir):
        shutil.rmtree(temp_wavtext_dir)
    pathlib.Path(temp_wavtext_dir).mkdir(parents=True)
    if os.path.exists(final_textgrid_dir):
        shutil.rmtree(final_textgrid_dir)
    pathlib.Path(final_textgrid_dir).mkdir(parents=True)

    temp_n = 100      # 100 files once forceAlignment
    aligns_l = []
    if os.path.isdir(wav_dir) and os.path.isdir(text_dir):
        print(f"start align {wav_dir} and {text_dir}")
        wavs = [os.path.join(wav_dir, w) for w in os.listdir(wav_dir) if w.endswith("wav")]
        txts = [os.path.join(text_dir, t.replace("wav", "txt")) for t in os.listdir(wav_dir) if t.endswith("wav")]

        iter_nums = int(len(wavs) / temp_n) + 1
        accumulate_nums = 0
        failed_files = []
        for i in range(iter_nums):
            print(f"{i}st / {iter_nums} iter force alignment started...")
            std = i * temp_n
            end = min((i + 1) * temp_n, len(wavs))
            accumulate_nums += end - std

            wav_sub, txts_sub = wavs[std : end], txts[std : end]
            # copy into wavtext_temp_dir
            [copyfile(w, os.path.join(temp_wavtext_dir, os.path.basename(w))) for w in wav_sub]
            [copyfile(t, os.path.join(temp_wavtext_dir, os.path.basename(t))) for t in txts_sub]
            # execute montreal in temp_wavtext_dir
            cmd = "{} -c -q {} {} {} {}".format(mon_args["executor"], temp_wavtext_dir,
                                                mon_args["lexicon"], mon_args["lang"], temp_textgrid_dir)
            print(cmd)
            print("montreal align start ...")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            (out, err) = proc.communicate()
            out = out.decode("utf-8")
            err_pattern = "error writing the TextGrid for (.*),"
            err_search = re.search(err_pattern, out)

            # due with failed-to-align wav/txt file by removing them from temp dir.
            while err_search is not None:
                err_f = err_search.group(1)
                print("Found failed-to-aligned file: {}".format(err_f))
                failed_files.append(err_f)
                # 1_1. remove error and aligned wav/txt file (prevent aligned again) from temp_wavtext_dir
                err_wt = [err_f + ".wav", err_f + ".txt"]
                if os.path.isdir(temp_textgrid_mnt_dir):
                    algd_f = [w[:w.index(".")] for w in os.listdir(temp_textgrid_mnt_dir) if w.endswith(".TextGrid")]
                    algd_w = [f + ".wav" for f in algd_f]
                    algd_t = [f + ".txt" for f in algd_f]
                    remove_file(temp_wavtext_dir, err_wt + algd_w + algd_t)
                else:
                    remove_file(temp_wavtext_dir, err_wt)
                # 1_2. copy aligned output
                if os.path.isdir(temp_textgrid_mnt_dir):
                    copy_tree(temp_textgrid_mnt_dir, final_textgrid_dir)
                # 1_3. re-alignment
                print("re-force-alignment started...")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()
                out = out.decode("utf-8")
                err_search = re.search(err_pattern, out)

            # 2. remove temp_file
            # copy aligned dir temp_textgrid_dir to final_textgrid_dir
            if os.path.isdir(temp_textgrid_mnt_dir):
                copy_tree(temp_textgrid_mnt_dir, final_textgrid_dir)
                print(f"successForced: {len(os.listdir(temp_textgrid_mnt_dir))} / iter_files: {end - std}")
                print(f"total_successForced : {len(os.listdir(final_textgrid_dir))} /total_iter_files: {accumulate_nums}")
                # remove temp_wavtext_dir
                for filename in os.listdir(temp_wavtext_dir):
                    f_p = os.path.join(temp_wavtext_dir, filename)
                    os.unlink(f_p)
            else:
                print(f"Nothing aligned in {temp_wavtext_dir}")
        print("All force alignment finished and {} textgrids files are generated".format(len(os.listdir(final_textgrid_dir))))
        copy_tree(final_textgrid_dir, output_textgird_dir)
    else:
        raise IOError("wav and text should be file or dir, but got {} {}".format(wav_dir, text_dir))
    return "", output_textgird_dir

def remove_file(temp_dir, files):
    for f in files:
        os.unlink(os.path.join(temp_dir, f))

if __name__ == '__main__':
    # test in ForceAligner instead
    align_wavs_montreal()