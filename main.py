
import os
import argparse
import sys
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BookBuddy AI - 智能书本问答系统')
    
    # 添加命令行选项
    parser.add_argument('--process', action='store_true', 
                        help='处理data/raw目录下的所有书籍')
    parser.add_argument('--embed', action='store_true',
                        help='为处理后的文本生成嵌入向量')
    parser.add_argument('--build-index', action='store_true',
                        help='构建向量存储索引')
    parser.add_argument('--ui', choices=['gradio', 'streamlit'], default='gradio',
                        help='选择用户界面类型 (默认: gradio)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Web界面端口 (默认: 7860)')
    parser.add_argument('--book', type=str, help='处理单本书籍的路径')
    
    return parser.parse_args()

def process_books(book_path=None):
    """处理书籍，包括提取、清洗和分割文本"""
    from src.data_processing.text_extraction import process_book, process_all_books
    from src.data_processing.text_cleaning import clean_and_save
    from src.data_processing.text_splitting import split_and_save, process_all_cleaned_texts
    
    logger.info("开始处理书籍...")
    
    # 处理单本书籍或所有书籍
    if book_path:
        logger.info(f"处理单本书籍: {book_path}")
        processed_file = process_book(book_path)
        if processed_file:
            clean_and_save(processed_file)
            split_and_save(processed_file)
    else:
        # 提取文本
        processed_files = process_all_books()
        
        # 清洗文本
        for file_path in processed_files:
            clean_and_save(file_path)
        
        # 分割文本
        process_all_cleaned_texts()
    
    logger.info("书籍处理完成")

def create_embeddings():
    """为处理后的文本生成嵌入向量"""
    from src.embeddings.embedding_generator import process_all_chunks
    
    logger.info("开始生成嵌入向量...")
    success = process_all_chunks()
    if success:
        logger.info("嵌入向量生成完成")
    else:
        logger.error("嵌入向量生成失败")

def build_vector_index():
    """构建向量存储索引"""
    from src.embeddings.vector_store import build_vector_store
    
    logger.info("开始构建向量索引...")
    result = build_vector_store()
    if result:
        logger.info(f"向量索引构建完成: {result}")
    else:
        logger.error("向量索引构建失败")

def launch_ui(ui_type='gradio', port=7860):
    """启动用户界面"""
    if ui_type == 'gradio':
        try:
            from ui.gradio_app import launch_app
            logger.info(f"启动Gradio界面，端口: {port}")
            launch_app(port=port)
        except ImportError:
            logger.error("无法导入Gradio。请确保已安装: pip install gradio")
            sys.exit(1)
    
    elif ui_type == 'streamlit':
        try:
            import streamlit
            logger.info("Streamlit界面无法直接从这里启动")
            logger.info(f"请使用以下命令运行Streamlit界面:")
            logger.info(f"streamlit run {os.path.join('ui', 'streamlit_app.py')}")
        except ImportError:
            logger.error("无法导入Streamlit。请确保已安装: pip install streamlit")
            sys.exit(1)
    
    else:
        logger.error(f"不支持的UI类型: {ui_type}")

def main():
    """主函数"""
    args = parse_args()
    
    # 处理书籍
    if args.process:
        process_books(args.book)
    
    # 生成嵌入向量
    if args.embed:
        create_embeddings()
    
    # 构建向量索引
    if args.build_index:
        build_vector_index()
    
    # 如果没有指定任何处理任务，则启动UI
    if not (args.process or args.embed or args.build_index):
        launch_ui(args.ui, args.port)

if __name__ == "__main__":
    main()