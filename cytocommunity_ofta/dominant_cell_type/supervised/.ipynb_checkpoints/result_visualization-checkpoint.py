import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sci_palettes
import os
import shutil
import datetime
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42     # make text in plot editable in AI.
#print(sci_palettes.PALETTES.keys())     # used for checking all color schemes of different journals.
#sci_palettes.PALETTES["d3_category20"]     # see detailed color code
sci_palettes.register_cmap("d3_category20")    # register a specific palette for TCN coloring.


def result_visualization(timestamp):
    
    ## Hyperparameters
    InputFolderName = "/home/owkin/project/cytocommunity_results/dominant_cell_type/raw/"
    
    
    ## Create output folders
    ThisStep_OutputFolderName = f"/home/owkin/project/cytocommunity_results/dominant_cell_type/supervised/experiments/{timestamp}/result_visualization/"
    if os.path.exists(ThisStep_OutputFolderName):
        shutil.rmtree(ThisStep_OutputFolderName)
    os.makedirs(ThisStep_OutputFolderName)
    
    OutputFolderName_1 = ThisStep_OutputFolderName + "TCN_Plot/"
    os.mkdir(OutputFolderName_1)
    OutputFolderName_2 = ThisStep_OutputFolderName + "CellType_Plot/"
    os.mkdir(OutputFolderName_2)
    OutputFolderName_3 = ThisStep_OutputFolderName + "ResultTable_File/"
    os.mkdir(OutputFolderName_3)
    
    
    ## Import image name list.
    Region_filename = InputFolderName + "ImageNameList.txt"
    region_name_list = pd.read_csv(
            Region_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["Image"],  # set our own names for the columns
        )
    
    ## Import the cell type list used for matching color palettes across different cell type plots.
    unique_cell_type_df = pd.read_csv(
            "/home/owkin/project/cytocommunity_results/dominant_cell_type/supervised/graphs/neighbors_order_1/UniqueCellTypeList.txt",
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["UniqueCellType"],  # set our own names for the columns
        )
    UniqueCellType_vec = unique_cell_type_df['UniqueCellType'].values.tolist()
    
    ## Initialize a TCN code list used for matching color palettes across different TCN plots.
    UniqueTCN_vec = list(range(1, 21))  # 20 TCNs at maximum.
    UniqueTCN_vec = [str(element) for element in UniqueTCN_vec]
    
    
    ## Traverse all images in the dataset.
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    LastStep_OutputFolderName = f"/home/owkin/project/cytocommunity_results/dominant_cell_type/supervised/experiments/{timestamp}/ensemble/"
    for graph_index in range(0, len(region_name_list)):
        
        print(f"This is Image{graph_index+1}/{len(region_name_list)}")
        
        region_name = region_name_list.Image[graph_index]
    
        ## Import target cellular spatial graph x/y coordinates.
        GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
        x_y_coordinates = pd.read_csv(
                GraphCoord_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["x_coordinate", "y_coordinate"],  # set our own names for the columns
            )
        target_graph_map = x_y_coordinates
        #target_graph_map["y_coordinate"] = 0 - target_graph_map["y_coordinate"]  # for consistent with original paper. Don't do this is also ok.
    
        ## Import cell type label.
        CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
        cell_type_label = pd.read_csv(
                CellType_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["cell_type"],  # set our own names for the columns
            )
        # Add cell type labels to target graph x/y coordinates.
        target_graph_map["Cell_Type"] = cell_type_label.cell_type
        # Below is for matching color palettes across different cell type plots, which is quite useful for supervised tasks.
        target_graph_map["Cell_Type"] = pd.Categorical(target_graph_map["Cell_Type"], UniqueCellType_vec)
    
        ## Import the final TCN labels to target graph x/y coordinates.
        MajorityVoting_FileName = LastStep_OutputFolderName + "ImageCollection/" + region_name + "/TCNLabel_MajorityVoting.csv"
        target_graph_map["TCN_Label"] = np.loadtxt(MajorityVoting_FileName, dtype='int', delimiter=",")
        # Converting integer list to string list for making color scheme discrete.
        target_graph_map.TCN_Label = target_graph_map.TCN_Label.astype(str)
        # Below is for matching color palettes across different TCN plots, which is quite useful for supervised tasks.
        target_graph_map["TCN_Label"] = pd.Categorical(target_graph_map["TCN_Label"], UniqueTCN_vec)
    
    
        #-----------------------------------------Generate plots-------------------------------------------------#
        ## Plot x/y map with "TCN_Label" coloring.
        TCN_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, hue="TCN_Label", palette="d3_category20", alpha=1.0, s=10.0, legend="full")   # 20 colors at maximum.
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
        TCN_fig_filename1 = OutputFolderName_1 + "TCN_" + region_name + ".pdf"
        plt.savefig(TCN_fig_filename1)
        TCN_fig_filename2 = OutputFolderName_1 + "TCN_" + region_name + ".png"
        plt.savefig(TCN_fig_filename2)
        plt.close()
    
    
        ## Plot x/y map with "Cell_Type" coloring.
        CellType_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map, hue="Cell_Type", palette=sns.color_palette("husl", 30), alpha=1.0, s=10.0, legend="full")  # 30 colors at maximum.
        # Hide all four spines
        CellType_plot.spines.right.set_visible(False)
        CellType_plot.spines.left.set_visible(False)
        CellType_plot.spines.top.set_visible(False)
        CellType_plot.spines.bottom.set_visible(False)
        CellType_plot.set(xticklabels=[])  # remove the tick label.
        CellType_plot.set(yticklabels=[])
        CellType_plot.set(xlabel=None)  # remove the axis label.
        CellType_plot.set(ylabel=None)
        CellType_plot.tick_params(bottom=False, left=False)  # remove the ticks.
        # Place legend outside top right corner of the CURRENT plot
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
        # Save the CURRENT figure.
        CellType_fig_filename1 = OutputFolderName_2 + "CellType_" + region_name + ".pdf"
        plt.savefig(CellType_fig_filename1)
        CellType_fig_filename2 = OutputFolderName_2 + "CellType_" + region_name + ".png"
        plt.savefig(CellType_fig_filename2)
        plt.close()
    
    
        ## Export result dataframe: "target_graph_map".
        TargetGraph_dataframe_filename = OutputFolderName_3 + "ResultTable_" + region_name + ".csv"
        target_graph_map.to_csv(TargetGraph_dataframe_filename, na_rep="NULL", index=False) #remove row index.
    
    
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

