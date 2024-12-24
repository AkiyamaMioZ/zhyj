import os
import django
import pandas as pd

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'steamDB.settings')
django.setup()

from app01.models import Game

def import_games_from_csv(csv_file):
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 遍历 DataFrame 并将数据插入到数据库中
    for index, row in df.iterrows():
        game = Game(
            game_id=row['game_id'],
            game_name=row['game_name'],
            release_date=row['release_date'],
            cur_price=row['cur_price'],
            last_price=row['last_price'],
            tags=row['tags'],
            steam_db_rating=row['steam_db_rating'],
            good_reviews_count=row['good_reviews_count'],
            negative_reviews_count=row['negative_reviews_count'],
            all_time_peak=row['all_time_peak'],
            page_url=row['page_url']
        )
        game.save()

    print("Data imported successfully")

if __name__ == '__main__':
    csv_file = r'D:\workspacePy\Y3项目\steamDB\app01\clear_data.csv'  # 替换为你的 CSV 文件路径
    import_games_from_csv(csv_file)