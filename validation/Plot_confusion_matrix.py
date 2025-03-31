'''
Created on 27.03.2025

@author: L.Reischmann
'''
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def capitalize_first_letter(input_string):
    if len(input_string) == 0:
        return input_string
    return input_string[0].upper() + input_string[1:].lower()

def load_and_process_cloud_data(v1_site, v2_site):
    """
    Loads cloud classification data from two different sources and processes them.

    Parameters:
    -----------
    - v1_site (str): Path to the cloud classification data Version 1 data file.
    - v2_site (str): Path to the cloud classification data Version 2 data file.
                     Depending on the version group name or variable might change 

    Returns:
    --------
    tuple: Contains two DataFrames with processed cloud classifications
    """
    # Load data for cloud classification algorithm Version 1 files
    cc1 = xr.open_dataset(v1_site, group='measurement_info')
    cloudarr1 = cc1['cloud_classification'].values
    cc1.close()
    
    # Load data for cloud classification algorithm Version 2 files
    cc2 = xr.open_dataset(v2_site, group='CLOUD')
    cloudarr2 = cc2['main_classification'].values
    cc2.close()
    
    # Process cloud classification data
    ccdf1 = process_cloud_classification(cloudarr1)
    ccdf2 = process_cloud_classification(cloudarr2)
    
    return ccdf1, ccdf2

def process_cloud_classification(cloudarr):
    '''
    This functions turns an array of cloud classification results to a
    pandas dataframe.
    
    Parameters:
    ------------
    - cloud classification results (np.array): cloud classification results
                                            

    Returns:
    ---------
    pandas dataframe: Contains cloud classifications results
    '''
    # Make sure that nan's are processed correctly
    cloudarr = np.nan_to_num(cloudarr, nan=0, posinf=0, neginf=0)
    cloudarr = cloudarr.astype(np.uint8)
    clear_low = (cloudarr & 128) // 128
    clear_high = (cloudarr & 64) // 64
    holes = (cloudarr & 32) // 32
    broken = (cloudarr & 16) // 16
    continuous = (cloudarr & 8) // 8
    cloud_classification_df = pd.DataFrame({
        'clear_sky_low': clear_low,
        'clear_sky_high': clear_high,
        'cloud_holes': holes,
        'broken_clouds': broken,
        'continuous_clouds': continuous,
        })
    return cloud_classification_df

def confusion_matrix_to_pdf(folder_v1, folder_v2, save_path, savepdf=True):
    """
    Generates confusion matrices from NetCDF files in two folders and saves them as a PDF.

    Parameters:
    ------------
    - folder_v1 : str  -- Path to the first folder of NetCDF (*.nc) files
    - folder_v2 : str  -- Path to the second folder of NetCDF (*.nc) files
    - save_path : str  -- Directory to save the PDF
    - savepdf : bool   -- If True, saves results to a PDF (default: True)

    Functionality:
    --------------
    - Matches files by extracting site names from filenames
    - Loads and processes cloud data (`load_and_process_cloud_data`)
    - Generates and saves confusion matrices (`plot_confusion_matrix`)
    - Skips mismatched files and prints status updates.  

    Returns:
    -------
    - A multi-page PDF (`Confusion_Matrix_all_sites.pdf`)
      in `save_path` (if `savepdf=True`).  
    """
    filelist_nc_v1 = glob.glob(os.path.join(folder_v1, '*.nc'))
    filelist_nc_v2 = glob.glob(os.path.join(folder_v2, '*.nc'))
    pdf_path = os.path.join(save_path, 'Confusion_Matrix_all_sites.pdf')
    if savepdf:
        print('Results are saved to: ', pdf_path)
    with PdfPages(pdf_path) as pdf_pages:
        ziped_filelist = zip(filelist_nc_v1, filelist_nc_v2)
        for _, (v1_site, v2_site) in enumerate(ziped_filelist):
            # Extract site names from file names
            site_name_v1 = v1_site.split('-')[4]
            site_name_v2 = v2_site.split('-')[4]
            
            # Check if site names match
            if site_name_v1 != site_name_v2:
                print(f"Site name mismatch: {site_name_v1} != {site_name_v2}")
                continue
            
            print('Currently handle: ', site_name_v1)
            cc_df1, cc_df2 = load_and_process_cloud_data(v1_site, v2_site)
            
            # Generate confusion matrix and save to PDF
            plot_confusion_matrix(cc_df1, cc_df2, save_path, site=site_name_v1,
                                  pdf_pages=pdf_pages, savepdf=savepdf)
                        

def plot_confusion_matrix(ccdf1, ccdf2, save_path, site, pdf_pages, savepdf=False):
    """
    This function computes and visualizes the confusion matrix between two
    cloud classification datasets. It generates a heatmap with the
    confusion matrix, including additional error analysis (commission, omission, 
    and overall accuracy).
    
    Parameters:
    ------------
    - ccdf1 (pd.DataFrame): DataFrame with true cloud classification labels.
    - ccdf2 (pd.DataFrame): DataFrame with predicted cloud classification labels.
    - save_path (str): Path to save the output plot.
    - pdf_pages: 
    - site (str): The name of the site or dataset.
    - savepdf (bool): If True, saves the plot as a PDF. If False, saves as PNG.
    """
    
    # Get the true and predicted labels for each sample
    true_labels = ccdf1.apply(lambda row: row.idxmax() if row.sum() > 0 else 'no_label', axis=1)
    pred_labels = ccdf2.apply(lambda row: row.idxmax() if row.sum() > 0 else 'no_label', axis=1)
    
    # Compute Confusion Matrix
    unique_labels = ccdf1.columns  # Use original column names as labels
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # Define label names for the confusion matrix plot
    labels = ['Clear sky, low aerosol','Clear sky, high aerosol',
              'Cloud Holes', 'Broken Clouds', 'Continuous Clouds']
    
    # Compute percentage-based confusion matrix
    total_sum = cm.sum()
    cm_percent_total = cm / total_sum * 100
    
    # Compute errors and accuracies
    commission_error = 100 - (np.diag(cm) / cm.sum(axis=1) * 100)
    commission_agreement = 100 - commission_error
    commission_sum = cm.sum(axis=1)
    
    omission_error = 100 - (np.diag(cm) / cm.sum(axis=0) * 100)
    omission_agreement = 100 - omission_error
    omission_sum = cm.sum(axis=0)
    
    overall_accuracy = np.trace(cm) / np.sum(cm) * 100
    
    # Set up the plot figure and axis
    fig, ax = plt.subplots(figsize=(9, 5))
    modern_palette = sns.color_palette("BuPu", as_cmap=True)
    heatmap_params = {
        'cmap': modern_palette,
        'ax': ax,
        'annot': False,
        'linewidths': 0,
        'annot_kws': {"fontsize": 10},
        'square': True,
        'fmt': 'd'
    }
    
    # Configure heatmap color bar
    heatmap_params['cbar_kws'] = {
        'shrink': 0.9,
        'pad': 0.125,
        'label': 'Percentage'
    }
    
    # Set heatmap axis limits and plot
    heatmap_params['vmin'] = 0
    heatmap_params['vmax'] = 100
    sns.heatmap(cm_percent_total, **heatmap_params)
    
    # Add numerical annotations to each cell in the heatmap
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f'{cm[i, j]}\n{cm_percent_total[i, j]:.2f}%'
            color = 'white' if cm_percent_total[i, j] > 50 else 'black'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=9, color=color)
    
    # Draw dashed lines between cells in the heatmap
    ax.hlines([1, 2, 3, 4, 5], *ax.get_xlim(), colors='grey',
              linestyles='dashed', lw=0.7)
    ax.vlines([1, 2, 3, 4, 5], *ax.get_ylim(), colors='grey',
              linestyles='dashed', lw=0.7)
    
    # Add bounding boxes around the diagonal
    diagonal_coordinates = [(i, i) for i in range(cm.shape[0])]
    for coord in diagonal_coordinates:
        bwid, blen = (1, 1)
        rect = FancyBboxPatch(coord, bwid, blen, boxstyle="round, pad=0",
                              linewidth=2, edgecolor='k', facecolor='None',
                              clip_on=False, transform=ax.transData, zorder=3)
        ax.add_patch(rect)

    # Plot lines for commission and omission errors
    lw_cat = 1.6
    c_bar = 'royalblue'
    alpha_bar = 1
    edge = len(ccdf1.columns)
    horizontal_lines = [1, 2, 3, 4, 5, 5.98]
    vertical_lines = [1, 2, 3, 4, 5]
    
    for y in horizontal_lines:
        ax.plot([edge, edge + 1.1], [y, y], color=c_bar, clip_on=False,
                lw=lw_cat, alpha=alpha_bar, zorder=5)
    
    for x in vertical_lines:
        ax.plot([x, x], [edge, edge + 0.8], color=c_bar, clip_on=False,
                lw=lw_cat, alpha=alpha_bar, zorder=5)
        
    ax.hlines(edge, 0, edge-1, colors=c_bar, lw=4, alpha=alpha_bar, zorder=5)
    ax.vlines(edge, 0, edge-1, colors=c_bar, lw=3.2, alpha=alpha_bar, zorder=5)
    
    # Annotations for commission and omission error statistics
    lift = 8
    ax.annotate('Commission', xy=(edge, 0), 
                xytext=(30, 12+lift), textcoords='offset points', ha='center',
                va='center', fontsize=8)
    ax.annotate('error', xy=(edge, 0), 
                xytext=(30, 4+lift), textcoords='offset points', ha='center',
                va='center', fontsize=8, color='red')
    
    ax.annotate('agreement', xy=(edge, 0), 
                xytext=(30, -4+lift), textcoords='offset points', ha='center',
                va='center', fontsize=8, color='green')
    
    ax.annotate('(sum)', xy=(edge, 0), 
                xytext=(30, -12+lift), textcoords='offset points', ha='center',
                va='center', fontsize=8, color='k')
    
    # Annotations for omission error statistics
    olift = 15
    hlift = -30
    ax.annotate('Omission', xy=(0, edge),
                xytext=(hlift, -24+olift), textcoords='offset points',
                ha='center', va='center', fontsize=8)
    ax.annotate('error', xy=(0, edge), color='red',
                xytext=(hlift, -32+olift), textcoords='offset points',
                ha='center', va='center', fontsize=8)
    ax.annotate('agreement', xy=(0, edge), color='green',
                xytext=(hlift, -40+olift), textcoords='offset points',
                ha='center', va='center', fontsize=8)
    ax.annotate('(sum)', xy=(0, edge), 
                xytext=(hlift, -48+olift), textcoords='offset points',
                ha='center', va='center', fontsize=8)
    
    # Display errors and agreements for each label
    n_rows = len(commission_error)
    y_coords_com = np.linspace(1, n_rows, n_rows)
    y_coords_om = np.linspace(1, n_rows, n_rows)
    # Coordinates for visualization 
    right_shift = 30
    com_lift = 28
    left_shift = -23
    om_lift = -13
    # Annotate the Errors 
    zipped_errors = zip(commission_error, omission_error)
    for i, (err_comm, err_omiss) in enumerate(zipped_errors):
        y_coord_com = y_coords_com[i]
        y_coord_om = y_coords_om[i]
        
        if not np.isnan(err_comm):
            ax.annotate(f'{err_comm:.2f}%', xy=(edge, y_coord_com), 
                        xytext=(right_shift, com_lift),
                        textcoords='offset points', ha='center', va='center',
                        fontsize=8, color='red')
            ax.annotate(f'\n{commission_agreement[i]:.2f}%',
                        xy=(edge, y_coord_com), ha='center', va='center',
                        xytext=(right_shift, com_lift-5), color='green',
                        textcoords='offset points', fontsize=8)
        ax.annotate(f'\n({commission_sum[i]})', xy=(edge, y_coord_com), 
                    xytext=(right_shift, com_lift-15), ha='center', color='k',
                    textcoords='offset points', va='center', fontsize=8)
    
        if not np.isnan(err_omiss):
            ax.annotate(f'{err_omiss:.2f}%', xy=(y_coord_om, edge), 
                        xytext=(left_shift, om_lift), ha='center', va='center',
                        textcoords='offset points',  fontsize=8, color='red')
            ax.annotate(f'\n{omission_agreement[i]:.2f}%', ha='center',
                        xy=(y_coord_om, edge), xytext=(left_shift, om_lift-5),
                        textcoords='offset points',  va='center', fontsize=8,
                        color='green')
        ax.annotate(f'\n({omission_sum[i]})', xy=(y_coord_om, edge), 
                    xytext=(left_shift, om_lift-15), textcoords='offset points',
                    ha='center', va='center', fontsize=8, color='k')
    
    # Text of total overall accuracy
    v_lift = 0.25
    ax.text(edge+0.67, n_rows + v_lift, f'Overall\naccuracy', 
            ha='center', va='center', fontsize=6, color='k')
    ax.text(edge+0.67, n_rows + v_lift+0.35, f'\n{np.trace(cm)} / {np.sum(cm)}', 
            ha='center', va='center', fontsize=8, color='k')
    ax.text(edge+0.67, n_rows + v_lift+0.19, f'\n{overall_accuracy:.2f}%', 
            ha='center', va='center', fontsize=8, color='green')
    
    site = capitalize_first_letter(site)
    ax.text(-1.4, -0.5 , f'{site}', ha='center',
            va='center', weight='bold', fontsize=10, color='k')
    
    # Label the axes and set ticks
    ax.xaxis.set_ticks_position("top")
    ax.set_xlabel('Cloud Classification ' + r'Version$\mathbf{\ 2}$',
                  labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel('Cloud Classification ' + r'Version$\mathbf{\ 1}$',
                  labelpad=10)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    
    # Adjust label positions
    n = 2
    offset = 0.05
    for i, label in enumerate(ax.get_xticklabels()):
        if i % n == 0:
            label.set_y(label.get_position()[1] + offset)
    
    # Save the plot as a PDF or PNG
    if savepdf:
        pdf_pages.savefig(fig, dpi=600, bbox_inches='tight')
    else:
        png_path = os.path.join(save_path, f'{site}_V1_v2_confusionmatrix.png')
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        print(f'Plot saved to: {png_path}')    
    # Close the plot
    plt.close(fig)


if __name__ == "__main__":
    # Define paths for the validation data folders '.nc' files needs to be in
    # this folder
    folder_v1 = r'D:\lreischmann\projects\FRM4DOAS\CloudCLassification_V2\validation_data_v1'
    folder_v2 = r'D:\lreischmann\projects\FRM4DOAS\CloudCLassification_V2\validation_data_v2'
    
    # Set the save path for the output PDF
    save_path = os.path.dirname(folder_v1)

    # Plot Matrix and save to pdf if savepdf=True, else False and saved to png
    confusion_matrix_to_pdf(folder_v1, folder_v2, save_path, savepdf=False)
    