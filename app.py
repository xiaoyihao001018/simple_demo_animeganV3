# 导入必要的库
from flask import Flask, request, send_file, render_template  # Flask Web框架相关组件
import onnxruntime as ort  # ONNX模型运行时
import cv2  # OpenCV图像处理库
import numpy as np  # 数值计算库
import io  # 处理二进制流
import os  # 操作系统接口
import logging  # 日志模块

# 配置日志系统
logging.basicConfig(level=logging.INFO)  # 设置日志级别为INFO
logger = logging.getLogger(__name__)  # 获取当前模块的logger

# 创建Flask应用实例
app = Flask(__name__)  # __name__是当前模块名

def process_image(img, model_name):
    """
    预处理图片函数
    参数:
        img: OpenCV格式的图片
        model_name: 模型名称
    返回:
        处理后的图片
    """
    h, w = img.shape[:2]  # 获取图片高度和宽度

    def to_8s(x):
        # 将尺寸调整为8的倍数，如果小于256则返回256
        return 256 if x < 256 else x - x % 8

    # 调整图片大小为8的倍数
    img = cv2.resize(img, (to_8s(w), to_8s(h)))
    # BGR转RGB并归一化到[-1,1]范围
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

def convert_image(image_data, model):
    """
    转换图片风格的主函数
    参数:
        image_data: 原始图片数据
        model: 加载的ONNX模型
    返回:
        转换后的图片数据流
    """
    try:
        # 将二进制图片数据转换为OpenCV格式
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 预处理图片
        processed_img = process_image(img, "AnimeGANv3_Hayao_36.onnx")
        processed_img = np.expand_dims(processed_img, axis=0)  # 添加batch维度

        # 使用模型进行推理
        input_name = model.get_inputs()[0].name  # 获取模型输入节点名
        fake_img = model.run(None, {input_name: processed_img})  # 执行模型推理

        # 后处理：将输出转换回正常图片格式
        fake_img = (np.squeeze(fake_img[0]) + 1.) * 127.5  # 反归一化
        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)  # 裁剪到[0,255]范围
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)  # RGB转BGR

        # 将图片编码为PNG格式的字节流
        _, buffer = cv2.imencode('.png', fake_img)
        return io.BytesIO(buffer.tobytes())
    except Exception as e:
        logger.error(f"图片处理失败: {str(e)}")
        raise

def load_model(model_path):
    """
    加载ONNX模型
    参数:
        model_path: 模型文件路径
    返回:
        加载好的模型会话
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    try:
        # 创建ONNX运行时会话，使用CPU推理
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        logger.info("模型加载成功！")
        return session
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

@app.route('/')
def index():
    """首页路由处理函数"""
    return render_template('index.html')  # 渲染index.html模板

@app.route('/convert', methods=['POST'])
def convert():
    """
    处理图片转换请求的路由
    接受POST请求，包含图片文件
    返回转换后的图片或错误信息
    """
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return {'error': '没有上传图片'}, 400

        file = request.files['image']
        if file.filename == '':
            return {'error': '未选择文件'}, 400

        # 检查文件类型是否支持
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return {'error': '不支持的文件类型'}, 400

        # 读取图片数据
        image_data = file.read()
        if not image_data:
            return {'error': '文件为空'}, 400

        # 处理图片并记录日志
        logger.info(f"开始处理图片，大小: {len(image_data)} bytes")
        result = convert_image(image_data, model)
        logger.info("图片处理完成")

        # 返回处理后的图片
        return send_file(
            result,
            mimetype='image/png',
            as_attachment=True,
            download_name='anime_style.png'
        )

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return {'error': str(e)}, 500

# 程序入口点
if __name__ == '__main__':
    try:
        # 加载模型
        model = load_model("models/AnimeGANv3_Hayao_36.onnx")
        # 启动Flask服务器，监听所有地址，端口8602
        app.run(host='0.0.0.0', port=8602, debug=True)
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")