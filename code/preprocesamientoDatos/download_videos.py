import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yt_dlp
import os

def download_video():
    url = url_entry.get()
    if not url:
        messagebox.showerror("Error", "Por favor, ingrese una URL de YouTube")
        return
    
    folder = filedialog.askdirectory()
    if not folder:
        return
    
    try:
        format_choice = format_var.get()
        
        if format_choice == "MP3":
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': f'{folder}/%(title)s.%(ext)s',
                'verbose': False
            }
        else:  # MP4
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': f'{folder}/%(title)s.%(ext)s',
                'verbose': False
            }
            
        # Mostrar progreso
        status_label.config(text="Descargando...")
        root.update()
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            ydl.download([url])
        
        status_label.config(text="")
        messagebox.showinfo("Éxito", f"Descarga completada en {folder}\nArchivo: {title}.{format_choice.lower()}")
        
    except Exception as e:
        status_label.config(text="")
        messagebox.showerror("Error", f"Ocurrió un error: {e}")

# Crear la ventana
root = tk.Tk()
root.title("Descargador de YouTube")
root.geometry("450x250")

# Etiqueta y entrada de URL
tk.Label(root, text="URL del video:").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# Selector de formato
format_frame = tk.Frame(root)
format_frame.pack(pady=10)

tk.Label(format_frame, text="Formato:").pack(side=tk.LEFT)
format_var = tk.StringVar(value="MP4")
format_options = ttk.Combobox(format_frame, textvariable=format_var, values=["MP3", "MP4"], width=10, state="readonly")
format_options.pack(side=tk.LEFT, padx=10)

# Etiqueta de estado
status_label = tk.Label(root, text="", fg="blue")
status_label.pack(pady=5)

# Botón para descargar
download_button = tk.Button(root, text="Descargar", command=download_video, bg="#4CAF50", fg="white", width=15, height=2)
download_button.pack(pady=15)

# Agregar información de uso
tk.Label(root, text="MP3: Solo audio | MP4: Audio + Video", fg="gray").pack(pady=5)

root.mainloop()