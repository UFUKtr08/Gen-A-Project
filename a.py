import psutil

mem = psutil.virtual_memory()
swap = psutil.swap_memory()

print(f"Toplam RAM       : {mem.total / 1e9:.2f} GB")
print(f"Kullanılan RAM   : {mem.used / 1e9:.2f} GB")
print(f"Kullanılabilir RAM: {mem.available / 1e9:.2f} GB")
print()
print(f"Toplam Swap      : {swap.total / 1e9:.2f} GB")
print(f"Kullanılan Swap  : {swap.used / 1e9:.2f} GB")
print(f"Boş Swap         : {swap.free / 1e9:.2f} GB")
