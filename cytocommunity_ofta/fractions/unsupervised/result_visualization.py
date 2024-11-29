import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sci_palettes
import os
import shutil
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42     # make text in plot editable in AI.
#print(sci_palettes.PALETTES.keys())     # used for checking all color schemes of different journals.
#sci_palettes.PALETTES["d3_category20"]     # see detailed color code
sci_palettes.register_cmap("d3_category20")    # register a specific palette for TCN coloring.


def result_visualization(Image_Name, timestamp, InputFolderName):
    ## Hyperparameters
    InputFolderName = "/home/owkin/project/cytocommunity_results/fractions/raw/"
    
    
    ## Import target cellular spatial graph x/y coordinates.
    GraphCoord_filename = InputFolderName + Image_Name + "_Coordinates.txt"
    x_y_coordinates = pd.read_csv(
            GraphCoord_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["x_coordinate", "y_coordinate"],  # set our own names for the columns
        )
    target_graph_map = x_y_coordinates
    #target_graph_map["y_coordinate"] = 0 - target_graph_map["y_coordinate"]  # for consistent with original paper. Don't do this is also ok.

    ## Import the final TCN labels to target graph x/y coordinates.
    LastStep_OutputFolderName = "/home/owkin/project/cytocommunity_results/fractions/experiments/" + Image_Name + "/" + timestamp + "/ensemble/"
    target_graph_map["TCN_Label"] = np.loadtxt(LastStep_OutputFolderName + "TCNLabel_MajorityVoting.csv", dtype='int', delimiter=",")
    # Converting integer list to string list for making color scheme discrete.
    target_graph_map.TCN_Label = target_graph_map.TCN_Label.astype(str)
    
    
    #-----------------------------------------Generate plots-------------------------------------------------#
    ThisStep_OutputFolderName = "/home/owkin/project/cytocommunity_results/fractions/experiments/" + Image_Name + "/" + timestamp + "/result_visualization/"
    if os.path.exists(ThisStep_OutputFolderName):
        shutil.rmtree(ThisStep_OutputFolderName)
    os.makedirs(ThisStep_OutputFolderName)
    
    
    ## Plot x/y map with "TCN_Label" coloring.
    TCN_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, hue="TCN_Label", palette="d3_category20", alpha=1.0, s=20.0, legend="full")   # 20 colors at maximum.
    # Hide all four spines
    TCN_plot.spines.right.set_visible(False)
    TCN_plot.spines.left.set_visible(False)
    TCN_plot.spines.top.set_visible(False)
    TCN_plot.spines.bottom.set_visible(False)
    TCN_plot.set(xticklabels=[])  # remove the tick label.
    TCN_plot.set(yticklabels=[])
    TCN_plot.set(xlabel=None)  # remove the axis label.
    TCN_plot.set(ylabel=None)
    TCN_plot.tick_params(bottom=False, left=False)  # remove the ticks.
    # Place legend outside top right corner of the CURRENT plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    # Save the CURRENT figure.
    TCN_fig_filename1 = ThisStep_OutputFolderName + "TCN_" + Image_Name + ".pdf"
    plt.savefig(TCN_fig_filename1)
    TCN_fig_filename2 = ThisStep_OutputFolderName + "TCN_" + Image_Name + ".png"
    plt.savefig(TCN_fig_filename2)
    plt.close()
    
    
    ## Export result dataframe: "target_graph_map".
    TargetGraph_dataframe_filename = ThisStep_OutputFolderName + "ResultTable_" + Image_Name + ".csv"
    target_graph_map.to_csv(TargetGraph_dataframe_filename, na_rep="NULL", index=False) #remove row index.

