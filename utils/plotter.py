import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())

sorted_names = [name for hsv, name in by_hsv]       

_MUSCLE_LIST = ['triceps', 'biceps']

class Plotter():
    def __init__(self, width=8, height=6, nrows=3, ncols=2, max_simtime=4, headers=None):
        # configuration of figures
        font = {'family':'serif',
                'weight' : 'normal',
                'style': 'normal',
                'size'   : '10'}
        lines={'linewidth': '1',
                'linestyle': '-'}                
        axes = {'labelsize': 'medium',
                'titlesize': 'medium',
                'linewidth': '1',
                'grid': 'True',
                'facecolor': 'white',
                'edgecolor': 'k'}
        # pass in the dict as kwargs   
        plt.rc('font', **font)       
        plt.rc('lines', **lines)
        plt.rc('axes', **axes)                        

        min_x = 0
        max_x = max_simtime
        self.headers = headers
        plt.ion() 
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(width, height))
        

        # useful vectors: time, muscle's activation and angular position 

        self.time_buf = []     
        self.data_buf = dict(zip(self.headers, [[],[],[],[]]))
        self.mean_data_buf = dict(zip(_MUSCLE_LIST, [[],[]]))
        self.line_buf = {}
        self.mean_line_buf = {}     

        self.color_list = [colors['skyblue'], colors['salmon'], 'k-', 'g-'] 
        self.mean_color_list = dict(zip(_MUSCLE_LIST, [colors['steelblue'], colors['firebrick']]))
        self.min_y_list = dict(zip(self.headers, [-0.25, -0.25, -100, -0.25]))
        self.max_y_list = dict(zip(self.headers, [1.25, 1.25, 100, 1.25]))
        # plot tricep
        #self.ax[0,0].set_xlabel('Time (s)')
        #self.ax[0,0].set_ylabel(self.headers[0])   
        #self.line_buf[self.headers[0]] = self.ax[0,0].plot([], [], self.color_list[0])
        #self.ax[0,0].set_xlim((min_x,max_x))
        #self.ax[0,0].set_ylim((min_y,max_y))   
        #
        ## plot biceps
        #self.ax[0,1].set_xlabel('Time (s)')
        #self.ax[0,1].set_ylabel(self.headers[1])        
        #self.line_buf[self.headers[1]] = self.ax[0,1].plot([], [], self.color_list[1])
        #self.ax[0,1].set_xlim((min_x,max_x))
        #self.ax[0,1].set_ylim((min_y,max_y))   
        #
        ## plot distance
        #self.ax[1,0].set_xlabel('Time (s)')
        #self.ax[1,0].set_ylabel(self.headers[2])
        #self.line_buf[self.headers[2]] = self.ax[1,0].plot([], [], self.color_list[2])
        #
        ## plot reward
        #self.ax[1,0].set_xlabel('Time (s)')
        #self.ax[1,0].set_ylabel(self.headers[2])
        #self.line_buf[self.headers[2]] = self.ax[1,0].plot([], [], self.color_list[2])        

        for i in range(nrows):
            for j in range(ncols):
                name = self.headers[i*ncols + j]
                self.ax[i,j].set_xlabel('Time (s)')
                self.ax[i,j].set_title(name)
                self.ax[i,j].set_xlim((min_x, max_x))
                self.ax[i,j].set_ylim((self.min_y_list[name],self.max_y_list[name]))    
        
                self.line_buf[name] = self.ax[i,j].plot([], [], self.color_list[i*ncols + j])
                if name in _MUSCLE_LIST:
                    self.mean_line_buf[name] = self.ax[i,j].plot([], [], self.mean_color_list[name]) 
        
        # fancy plots
        self.fig.tight_layout()       
        
    def update_figure(self):
        for name in self.headers:
            line = self.line_buf[name][0] 
            line.set_xdata(self.time_buf)
            line.set_ydata(self.data_buf[name])

            if name in _MUSCLE_LIST:
                line = self.mean_line_buf[name][0] 
                line.set_xdata(self.time_buf)
                line.set_ydata(self.mean_data_buf[name])                

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def add_data (self, time, data):
        self.time_buf.append(time)
        for idx, name in enumerate(self.headers):
            self.data_buf[name].append(data[idx]) 
            if name in _MUSCLE_LIST:
                self.mean_data_buf[name].append(sum(self.data_buf[name][-10:])/len(self.data_buf[name][-10:]))


    def reset(self):
        self.time_buf = []     
        self.data_buf = dict(zip(self.headers, [[],[],[],[]]))  
        self.mean_data_buf = dict(zip(_MUSCLE_LIST, [[],[]]))
