from cmath import phase
from acpreprocessing.utils import io
import argschema
from argschema.fields import Str
import os
import matplotlib.pyplot as plt
import numpy as np

example_input = {
    "stitchDir": "/ACdata/processed/iSPIM2/MN8_S10_220211_high_res/stitch-s3/iter0",
    "filename": "pairwise-stitched.json",
    "d_name": "MN8_S10_220211_high_res",
    "save_dir": "/ACdata/processed/iSPIM2/MN8_S10_220211_high_res/stitch-s3/stitching_eval/"
}

def normalize(data):
    norm_data = []
    dmin, dmax = min(data), max(data)
    for i, val in enumerate(data):
        norm_data.append((val-dmin) / (dmax-dmin))
    return norm_data

class GraphStitchCorr(argschema.ArgSchema):
    stitchDir = Str(required=True, description='stitch iter directory')
    filename = Str(required=False, default="pairwise-stitched.json", description='which json file to grab stitch data from')
    d_name = Str(required=True, description='dataset name')

class GraphStitchCorr(argschema.ArgSchemaParser):
    default_schema = GraphStitchCorr

    def run(self):
        pairwise = io.read_json(os.path.join(self.args['stitchDir'], self.args["filename"]))
        n_pairs= len(pairwise)
        labels = []
        corrs = []
        variances = []
        displacement_z = []
        # displacement_x = []
        # displacement_y = []
        # phase_corrs = []
        for pair in pairwise:
            pair_label="{}&{}".format(pair[0]["tilePair"]["tilePair"][0]["index"],pair[0]["tilePair"]["tilePair"][1]["index"])
            labels.append(pair_label)
            variances.append(pair[0]["variance"])
            corrs.append(pair[0]["crossCorrelation"])
            # displacement_x.append(abs(pair[0]["displacement"][0]))
            # displacement_y.append(abs(pair[0]["displacement"][1]))
            displacement_z.append(abs(pair[0]["displacement"][2]))
            # phase_corrs.append(pair[0]["phaseCorrelation"])
        

        norm_var = normalize(variances)
        # norm_displ_x = normalize(displacement_x)
        # norm_displ_y = normalize(displacement_y)
        norm_displ_z = normalize(displacement_z)
        # norm_phase = normalize(phase_corrs)

        plt.plot(labels, corrs, linestyle='--', marker='x', label="cross correlation")
        plt.legend(loc="upper left")
        plt.xticks(rotation = 90)
        plt.xlabel("Tile Pair")
        plt.ylabel("Cross Correlation")
        plt.title("Cross correlation for {}".format(self.args['d_name']))
        plt.savefig(self.args["save_dir"]+'crossCorr.png', bbox_inches = "tight")

        plt.figure(2)
        plt.plot(labels, corrs, linestyle='--', marker='x', label="cross correlation")
        # plt.plot(labels, norm_displ_x, linestyle='--', marker='+', label="norm displacement x")
        # plt.plot(labels, norm_displ_y, linestyle='--', marker='+', label="norm displacement y")
        plt.plot(labels, norm_displ_z, linestyle='--', marker='+', label="norm displacement z")
        plt.legend(loc="upper left")
        plt.xticks(rotation = 90)
        plt.xlabel("Tile Pair")
        plt.ylabel("Values 0-1")
        plt.title("Cross correlation and Z Normalized Displacements for {}".format(self.args['d_name']))
        plt.savefig(self.args["save_dir"]+'with_z_displ.png', bbox_inches = "tight")

        plt.figure(3)
        plt.plot(labels, corrs, linestyle='--', marker='x', label="cross correlation")
        plt.plot(labels, norm_var, linestyle='--', marker='.', label="norm variance")
        plt.legend(loc="lower left")
        plt.xticks(rotation = 90)
        plt.xlabel("Tile Pair")
        plt.ylabel("Values 0-1")
        plt.title("Cross correlation and Normalized Variance for {}".format(self.args['d_name']))
        plt.savefig(self.args["save_dir"]+'with_norm_vars.png', bbox_inches = "tight")
        
        # plt.figure(4)
        # plt.plot(labels, corrs, linestyle='--', marker='x', label="cross correlation")
        # plt.plot(labels, phase_corrs, linestyle='--', marker='.', label="phase correlation")
        # plt.legend(loc="upper left")
        # plt.xticks(rotation = 90)
        # plt.xlabel("Tile Pair")
        # plt.ylabel("Values 0-1")
        # plt.title("Cross correlation and Phase Correlation for {}".format(self.args['d_name']))
        # plt.savefig(self.args["save_dir"]+'with_norm_phase.png', bbox_inches = "tight")


if __name__ == '__main__':
    mod = GraphStitchCorr(example_input)
    mod.run()