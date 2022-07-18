import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, width=5, height=6, nrows=3, ncols=2, max_simtime=4, headers=None):
        # configuration of figures
        font = {'family':'serif',
                'weight' : 'normal',
                'style': 'normal',
                'size'   : '12'}
        lines={'linewidth': '2',
                'linestyle': '-'}                
        axes = {'labelsize': 'small',
                'titlesize': 'small',
                'linewidth': '1',
                'grid': 'True',
                'facecolor': 'white',
                'edgecolor': 'k'}
        # pass in the dict as kwargs   
        plt.rc('font', **font)       
        plt.rc('lines', **lines)
        plt.rc('axes', **axes)                        
        min_y = -0.25
        max_y = 1.25
        min_x = 0
        max_x = max_simtime
        self.headers = headers
        plt.ion() 
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(width, height))
        self.fig.tight_layout()

        
        # useful vectors: time, muscle's activation and line 
        self.time_buf = []     
        self.act_buf = dict(zip(self.headers, [[],[],[],[],[],[],[]]))
        self.line_buf = {}      

        for i in range(nrows):
            for j in range(ncols):
                name = self.headers[i*ncols + j]
                self.ax[i,j].set_xlabel('Time (s)')
                self.ax[i,j].set_ylabel(name)
                self.ax[i,j].set_xlim((min_x,max_x))
                self.ax[i,j].set_ylim((min_y,max_y))    

                self.line_buf[name] = self.ax[i,j].plot([], [], 'r-')        
        
    def update_figure(self):
        for name in self.headers:
            line = self.line_buf[name][0] 
            line.set_xdata(self.time_buf)
            line.set_ydata(self.act_buf[name])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_data (self, time, act):
        self.time_buf.append(time)
        for idx, name in enumerate(self.headers):
            self.act_buf[name].append(act[idx]) 

    def reset(self):
        self.time_buf = []     
        self.act_buf = dict(zip(self.headers, [[],[],[],[],[],[],[]]))    
