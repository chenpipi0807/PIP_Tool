#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from pathlib import Path

# Check and install required packages
try:
    import pandas
    import openpyxl
except ImportError:
    import subprocess
    print("Installing required packages...")
    subprocess.call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
    import pandas

class ExcelToJsonConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel转JSON工具")
        self.root.geometry("600x300")
        self.root.configure(bg="#f0f0f0")
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        self.setup_ui()
        
        # 在Windows上使用简单方法模拟拖放功能
        # 告诉用户可以直接复制文件路径
        self.path_entry_focus_out = False

    def setup_ui(self):
        # 标题标签
        title_label = tk.Label(self.root, text="Excel转JSON工具", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # 说明标签
        instruction_label = tk.Label(self.root, text="请输入Excel文件路径或拖拽文件到此窗口", font=("Arial", 10), bg="#f0f0f0")
        instruction_label.pack(pady=10)
        
        # 文件路径输入框和浏览按钮框架
        path_frame = tk.Frame(self.root, bg="#f0f0f0")
        path_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(path_frame, textvariable=self.path_var, width=50)
        self.path_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        self.path_entry.bind("<FocusIn>", self.check_clipboard)
        
        browse_button = tk.Button(path_frame, text="浏览", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)
        
        # 转换按钮
        convert_button = tk.Button(self.root, text="转换", command=self.convert, width=20, height=2, 
                                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        convert_button.pack(pady=20)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        status_label = tk.Label(self.root, textvariable=self.status_var, fg="blue", bg="#f0f0f0")
        status_label.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel文件", "*.xlsx;*.xls")])
        if file_path:
            self.path_var.set(file_path)

    def check_clipboard(self, event=None):
        """Check clipboard for file path when entry gets focus"""
        try:
            clipboard_content = self.root.clipboard_get()
            if clipboard_content and os.path.exists(clipboard_content) and clipboard_content.lower().endswith(('.xlsx', '.xls')):
                self.path_var.set(clipboard_content)
                messagebox.showinfo("检测到Excel文件", f"已从剪贴板检测到Excel文件路径:\n{clipboard_content}")
        except:
            pass

    def convert(self):
        file_path = self.path_var.get().strip()
        
        if not file_path:
            messagebox.showerror("错误", "请选择Excel文件")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("错误", f"文件不存在: {file_path}")
            return
        
        try:
            self.status_var.set("正在转换...")
            self.root.update()
            
            # 读取Excel文件
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # 将DataFrame转换为字典列表，并添加批次ID
            data = []
            for i, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict['批次ID'] = str(i+1)  # 添加批次ID，从1开始
                data.append(row_dict)
            
            # 确定输出文件路径
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}.json"
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.status_var.set(f"转换完成！文件已保存至: {output_path}")
            messagebox.showinfo("成功", f"转换完成！文件已保存至:\n{output_path}")
            
        except Exception as e:
            self.status_var.set(f"转换失败: {str(e)}")
            messagebox.showerror("错误", f"转换失败: {str(e)}")

def main():
    # 创建主窗口
    root = tk.Tk()
    app = ExcelToJsonConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
