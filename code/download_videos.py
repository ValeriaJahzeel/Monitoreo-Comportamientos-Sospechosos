import tkinter as tk
from tkinter import filedialog, messagebox
import yt_dlp

def download_video():
    url = url_entry.get()
    if not url:
        messagebox.showerror("Error", "Por favor, ingrese una URL de YouTube")
        return
    
    folder = filedialog.askdirectory()
    if not folder:
        return
    
    try:   
        ydl_opts = {
        'outtmpl': f'{folder}/%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        messagebox.showinfo("Éxito", f"Descarga completada en {folder}")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {e}")

# Crear la ventana
root = tk.Tk()
root.title("Descargador de YouTube")
root.geometry("400x200")

# Etiqueta y entrada de URL
tk.Label(root, text="URL del video:").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# Botón para descargar
download_button = tk.Button(root, text="Descargar", command=download_video)
download_button.pack(pady=20)

root.mainloop()
