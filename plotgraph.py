from matplotlib.animation import ImageMagickWriter, FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image
# matplotlib.use("Agg")  # 好像用了之后图片plt.show不显示


class PlotGraph():
    def __init__(self, robot_num, thief_num, legend, mapfile=None) -> None:
        self.fps = 20
        self.robot_num = robot_num
        self.thief_num = thief_num
        colorstyle_list = ['k', 'g', 'r', 'c', 'm',
                           'y', 'peru', 'pink', 'purple', 'lime',
                           'cyan',   'gold', 'brown', 'gray', 'crimson',
                           'dodgerblue', 'yellow', 'royalblue']  # 颜色样式列表
        self.robotlines = []  # 画各机器人的Line2D列表
        self.thieflines = []  # 画小偷的Line2D列表
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0.05, 0.05, 0.75, 0.9])  # 左下角5%开始的90%的图
        self.time_text = self.ax.text(0, 1, '', transform=self.ax.transAxes)

        for color in colorstyle_list[:robot_num]:
            l, = self.ax.plot([], [], '-o', color=color)  # 生成各机器人的表示
            self.robotlines.append(l)
        for color in colorstyle_list[robot_num:thief_num+robot_num]:
            l, = self.ax.plot([], [], ':^', color=color)  # 生成小偷的表示
            self.thieflines.append(l)

        plt.legend(legend, bbox_to_anchor=(1.01, 0),
                   loc=3, borderaxespad=0)  # 图例
        plt.title('demo')

        self.ax.set_axis_off()  # 坐标轴关
        if mapfile:
            # 嵌入背景
            background = Image.open(mapfile)
            self.ax.imshow(background, extent=[
                           0, background.width, 0, background.height])

    def process_movedata(self, end_time, movedata: list):
        # 生成各帧的运动数据
        new_movedata = []
        t_list_ = list(np.arange(0, end_time+1/self.fps, 1/self.fps))
        for movedata_i in movedata:
            # 原数据
            t_list = movedata_i[0]
            x_list = movedata_i[1]
            y_list = movedata_i[2]
            # t_list_
            x_list_ = y_list_ = []
            # 线性插值
            x_list_ = np.interp(t_list_, t_list, x_list)
            y_list_ = np.interp(t_list_, t_list, y_list)
            # 新数据
            new_movedata.append([x_list_, y_list_])
            # plt.plot(x_list_, y_list_) 画轨迹线
        return t_list_, new_movedata

    def plotshow(self, end_time, movedata: list, thiefdata: dict, outpath, dpi=160):
        writer = FFMpegWriter(fps=self.fps)
        time_template = 'time = %.1fs'
        t_list, new_movedata = self.process_movedata(end_time, movedata)
        with writer.saving(self.fig, outpath, dpi=dpi):
            # 保存
            for index_t in range(len(t_list)):
                self.time_text.set_text(time_template % t_list[index_t])  # 写字
                # 机器人
                for j in range(self.robot_num):
                    l = self.robotlines[j]
                    x0, y0 = new_movedata[j][0][index_t], new_movedata[j][1][index_t]
                    l.set_data(x0, y0)
                    # 设置线的属性
                    # lines[0].set_linewidth(1)
                time = '%.2f' % t_list[index_t]
                # 小偷
                for j in range(self.thief_num):
                    if time in thiefdata[j]:
                        l = self.thieflines[j]
                        x0, y0 = thiefdata[j][time][0], thiefdata[j][time][1]
                        l.set_data(x0, y0)
                writer.grab_frame()


# if __name__ == '__main__':
#     # 生成7个随机的运动作为测试样例
#     """第i个机器人的
#     movedata[i]=[[t1,t2,t3],[x1,x2,x3],[y1,y2,y3]]
#     """
#     np.random.seed(19680801)

#     movedata = [[[], [], []] for _ in range(7)]
#     x0, y0 = 338, 337
#     points = [[x0, y0] for _ in range(7)]

#     for i in range(20):  # 20s
#         for j in range(7):
#             x0, y0 = points[j]
#             movedata[j][0].append(i)  # t
#             movedata[j][1].append(x0)  # x
#             movedata[j][2].append(y0)  # y
#             x0 += 20 * np.random.randn()
#             y0 += 20 * np.random.randn()
#             points[j] = x0, y0

#     # plotgraph = PlotGraph(robot_num=7, mapfile='.\grid-graph.png')  #
#     # plotgraph.plotshow(end_time=20, movedata=movedata,
#     #                    outpath='demo.mp4', dpi=240)
