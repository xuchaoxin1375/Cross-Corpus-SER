def f1():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一些随机数据
    x = np.random.rand(100)
    y = 3 * x + 2 + np.random.randn(100) * 0.5

    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                loss='mse')

    # 实时绘制训练过程
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], 'b-')

    for epoch in range(50):
        # 训练模型
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # 实时更新图表
        line.set_xdata(np.append(line.get_xdata(), epoch))
        line.set_ydata(np.append(line.get_ydata(), loss))
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
