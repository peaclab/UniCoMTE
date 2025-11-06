#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil 


def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg = 1.8, time_ticks =0.2):
    ax.set_xticks(np.arange(0,11,time_ticks))    
    ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),1.0))

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    ax.minorticks_on()
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax.plot(x,y, linewidth=lwidth)


lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def plot_12(
        ecg, 
        sample_rate = 500, 
        title       = 'ECG 12', 
        lead_index  = lead_index, 
        lead_order  = None,
        columns     = 2,
        speed = 50,
        voltage = 20,
        line_width = 0.6
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        speed      : signal speed on display, defaults to 50 mm / sec
        voltage    : signal voltage on display, defaults to 20 mm / mV
        line_width : line width, default to 0.6
    """
    if not lead_order:
        lead_order = list(range(0,len(ecg)))

    leads = len(lead_order)
    seconds = len(ecg[0])/sample_rate

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(
        ceil(len(lead_order)/columns),columns,
        sharex=True, 
        sharey=True,
        figsize=((speed/25)*seconds*columns,    # 1 inch= 25,4 mm. Rounded to 25 for simplicity
            (4.1*voltage/25)*leads/columns)     # 1 subplot usually contains values in range of (-2,2) mV
        )
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.06,  # the bottom of the subplots of the figure
        top    = 0.95
        )
    fig.suptitle(title)

    step = 1.0/sample_rate

    for i in range(0, len(lead_order)):
        if(columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i//columns,i%columns]
        t_lead = lead_order[i]
        t_ax.set_ylabel(lead_index[t_lead])
        t_ax.tick_params(axis='x',rotation=90)
       
        _ax_plot(t_ax, np.arange(0, len(ecg[t_lead])*step, step), ecg[t_lead], seconds)

def plot(
        ecg, 
        distractor,
        explanation,
        filename,
        sample_rate    = 500, 
        title          = 'Standard 12 Lead Electrocardiogram Recording', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        distractor : m x n ECG signal data that serves as distractor sample
        explanation: np.array (dx1)  
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    # define parameters
    separation_btw_cols = 1
    buffer_btw_title = 1.5
    separation_btw_rows = 7
    offset_shift = -4  # Adjust this value for desired separation

    print('ECG Counterfactual Plot:')
    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 1


    # create subplots
    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor + 8))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    # define mins and maxes
    
    x_min = 0
    x_max = columns*secs + separation_btw_cols
    y_min = row_height/4 - (rows/2)*row_height - buffer_btw_title - 0.5 - separation_btw_rows*rows #- 2.5
    y_max = row_height/4 + 1.5 + 2.0 
    print(y_min)
    print(y_max)


    # define colors
    print('defining styles alpha')
    if (style == 'bw'):
        print('else')
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
        distractor_line_color = (1,0,0)
    else:
        print('else')
        color_major = (0,0,0)
        color_minor = (0,0,0)
        color_line  = (0,0,1)
        distractor_line_color = (1,0,0)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major,alpha=0.4)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor, alpha = 0.3)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)
    ax.set_xticks(np.arange(x_min, x_max, 1.0))  # Less frequent major ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))  # Maintain minor ticks for grid
    ax.set_xlabel("Time", fontsize=24)
    ax.set_ylabel("Voltage",fontsize=24)
    fig.suptitle(title, fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

    # loop for plotting
    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows) - buffer_btw_title - separation_btw_rows*i
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25


                x_offset = 0
                if(c > 0):
                    x_offset = secs * c + separation_btw_cols
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)

         
                t_lead = lead_order[c * rows + i]
                
                # ADDED: offset to ensure mean is at zero
                lead_signal_shifted = ecg[t_lead] - np.mean(ecg[t_lead])
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset + 1, lead_index[t_lead], fontsize=20 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
                    lead_signal_shifted + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )
                
                #ADDED: Overlay distractor if the feature is in the explanation
                if t_lead in explanation:

                    # add offset so signals are not overlaid
                    

                    print('overlaying counterfactual with offset shift') 
                    
                    dist_signal_shifted = distractor[t_lead] - np.mean(distractor[t_lead])
         
                    #if(show_lead_name):
                        #ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=20 * display_factor)

                    if(show_lead_name):
                        ax.text(x_offset + 0.07, 
                                y_offset + 1 + offset_shift, 
                                lead_index[t_lead], 
                                fontsize=20 * display_factor,
                                color=(1, 0, 0))

                    ax.plot(
                        np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
                        dist_signal_shifted + y_offset + offset_shift,
                        linewidth=line_width * display_factor, 
                        color=distractor_line_color
                        )
                
                
                ### legends and lables
                from matplotlib.lines import Line2D

                # Define legend elements
                legend_elements = [
                    Line2D([0], [0], color=color_line, lw=2, label='Sample'),
                    Line2D([0], [0], color=distractor_line_color, lw=2, label='Explanation')
                ]

                # Add legend to the figure
                fig.legend(
                    handles=legend_elements,
                    loc='upper right',
                    ncol=1,                      # One column for a cleaner vertical stack
                    fontsize=20,
                    frameon=True,
                    framealpha=0.8,
                    edgecolor='gray',
                    facecolor='white'
                )

    # display and save
    print('saving')
    plt.savefig(filename)

                # # Add grid spacing annotation
                # ax.text(
                #     x_max - 2.5,                # Slightly inset from edge
                #     y_max - 0.2,
                #     "Each major grid: 0.2s × 0.5mV",
                #     fontsize=20,
                #     ha='right',
                #     va='top',
                #     bbox=dict(
                #         facecolor='white',
                #         alpha=0.8,
                #         edgecolor='gray',
                #         boxstyle='round,pad=0.3'
                #     )
                # )

                # # explanation annotation
                # ax.text(
                #     x_max - 21.4,                # Slightly inset from edge
                #     y_max - 0.15,
                #     "If the red signals replace the black signals they overlay,\nthe 12 lead ECG sample will be classified as \n$\mathbf{Normal}$ instead of $\mathbf{Sinus\ Bradycardia}$",
                #     fontsize=20,
                #     ha='left',
                #     va='top',
                #     bbox=dict(
                #         facecolor='white',
                #         alpha=0.8,
                #         edgecolor='gray',
                #         boxstyle='round,pad=0.3'
                #     )
                # )

def plot_1(ecg, sample_rate=500, title = 'ECG', fig_width = 15, fig_height = 2, line_w = 0.5, ecg_amp = 1.8, timetick = 0.2):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        fig_width  : The width of the plot
        fig_height : The height of the plot
    """
    plt.figure(figsize=(fig_width,fig_height))
    plt.suptitle(title)
    plt.subplots_adjust(
        hspace = 0, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.2,   # the bottom of the subplots of the figure
        top    = 0.88
        )
    seconds = len(ecg)/sample_rate

    ax = plt.subplot(1, 1, 1)
    #plt.rcParams['lines.linewidth'] = 5
    step = 1.0/sample_rate
    _ax_plot(ax,np.arange(0,len(ecg)*step,step),ecg, seconds, line_w, ecg_amp,timetick)
    
DEFAULT_PATH = './'
show_counter = 1
def show_svg(tmp_path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        tmp_path: path for temporary saving the result svg file
    """ 
    global show_counter
    file_name = tmp_path + "show_tmp_file_{}.svg".format(show_counter)
    plt.savefig(file_name)
    os.system("open {}".format(file_name))
    show_counter += 1
    plt.close()

def show():
    plt.show()


def save_as_png(file_name, path = DEFAULT_PATH, dpi = 100, layout='tight'):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
        dpi      : set dots per inch (dpi) for the saved image
        layout   : Set equal to "tight" to include ax labels on saved image
    """
    plt.ioff()
    plt.savefig(path + file_name + '.png', dpi = dpi, bbox_inches=layout)
    plt.close()

def save_as_svg(file_name, path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.svg')
    plt.close()

def save_as_jpg(file_name, path = DEFAULT_PATH):
    # not working for some reason.... 
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.jpg')
    plt.close()
