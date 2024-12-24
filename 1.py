import base64
import os

def save_base64_image(base64_string, filename):
    # 移除data:image/png;base64,前缀
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # 确保目录存在
    os.makedirs('static/images/icons', exist_ok=True)
    
    # 解码并保存图片
    with open(f'static/images/icons/{filename}', 'wb') as f:
        f.write(base64.b64decode(base64_string))

# 保存评论图标
comments_icon = """iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADDSURBVDiNY2AYBYMRMDIyMjAyMjL8//+f4f///wwMDAwM//79Y/j37x/D////Gf78+cPw9+9fhj9//jD8/v2b4c+fPwwMDAwMTAyoAGQAExMTAxMTE8P///8Z/v//z8DIyMjw//9/BkZGRgYmJiYGJiYmBmZmZgZmZmYGFhYWBhYWFgZWVlYGVlZWBjY2NgY2NjYGdnZ2Bg4ODgYODg4GTk5OBk5OTgYuLi4GLi4uBm5ubgZubm4GHh4eBh4eHgZeXl4G3lELhzsAABhxJKcyL6g5AAAAAElFTkSuQmCC"""

rate_up_icon = """iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADJSURBVDiNY2AYBaMAABMDAwPD////Gf7//8/AyMjI8P//fwZGRkYGJiYmBiYmJgZmZmYGZmZmBhYWFgYWFhYGVlZWBlZWVgY2NjYGNjY2BnZ2dgYODg4GDg4OBk5OTgZOTk4GLi4uBi4uLgZubm4Gbm5uBh4eHgYeHh4GXl5eBl5eXgY+Pj4GPj4+Bn5+fgZ+fn4GAQEBBgEBAQZBQUEGQUFBBiEhIQYhISEGYWFhBmFhYQYREREGERERBjExMQYxMTEGcXFxBvFRCwc7AAD/zySnVXJKHwAAAABJRU5ErkJggg=="""

rate_down_icon = """iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEwAACxMBAJqcGAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAADJSURBVDiNY2AYBQMKxMXFGcTFxRkkJCQYJCQkGKSkpBikpKQYZGRkGGRkZBjk5OQY5OTkGBQUFBgUFBQYlJSUGJSUlBhUVFQYVFRUGNTU1BjU1NQYNDTwG8DExMTAxMTEwMjIyPD//3+G////MzAwMDD8+/eP4d+/fwx//vxh+PPnD8Pv378Zfv/+zfDnzx+G379/M/z584fhz58/DH///mX4+/cvw79//xj+/fvH8P//f4b///8zMDIyMvz//5+BkZGRgYmJiYFp1MLBDQD/zySnVXJKHwAAAABJRU5ErkJggg=="""

save_base64_image(comments_icon, 'icon_comments.png')
save_base64_image(rate_up_icon, 'icon_rate_up.png')
save_base64_image(rate_down_icon, 'icon_rate_down.png')