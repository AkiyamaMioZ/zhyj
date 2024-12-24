
from django.contrib import admin
from django.urls import path
from app01.dnnRecommend import *
from app01.views import *

urlpatterns = [
    path('', show_display, name='show_display'),

    path('steamDB/login/', login, name='login'),
    path('steamDB/home/', home, name='home'),
    path('steamDB/show_display/', show_display, name='show_display'),
    path('steamDB/analysis/', analysis, name='analysis'),
    path('steamDB/recommendation/', recommendation, name='recommendation'),
    path('steamDB/recommendation/submit_comment/', submit_comment, name='submit_comment'),
    path('steamDB/company/', company, name='company'),
    path('steamDB/about/', about, name='about'),

    path('steamDB/api/register/', register_api, name='register_api'),
    path('steamDB/api/login/', login_api, name='login_api'),
    path('steamDB/api/logout/', logout_api, name='logout_api'),
    path('steamDB/api/get_all_api/', get_all_api, name='get_all_api'),
    path('steamDB/api/get_recommend_api/', get_recommend_api, name='get_recommend_api'),
    path('steamDB/api/by_tag/', get_games_by_tag, name='get_games_by_tag'),
    path('steamDB/api/submit_rating/', submit_rating, name='submit_rating'),
    path('steamDB/api/get_game_rating/', get_game_rating, name='get_game_rating'),
    path('steamDB/api/get_user_ratings/', get_user_ratings, name='get_user_ratings'),

    # 绘图
    path('steamDB/api/tag_statistics/', get_tag_statistics, name='tag_statistics'),
    path('steamDB/api/popularity_statistics/', get_popularity_statistics, name='popularity_statistics'),
    path('steamDB/api/price_statistics/', get_price_statistics, name='price_statistics'),
    path('steamDB/api/positive_rate_statistics/', get_positive_rate_statistics, name='positive_rate_statistics'),

    path('steamDB/api/positive_rate_price_correlation/', get_positive_rate_price_correlation,
         name='positive_rate_price_correlation'),
    path('steamDB/api/reviews_popularity_correlation/', get_reviews_popularity_correlation,
         name='reviews_popularity_correlation'),

    path('community/', post_list, name='post_list'),
    path('post/<int:post_id>/', post_detail, name='post_detail'),
    path('post/<int:post_id>/comment/', add_comment, name='add_comment'),
    path('post/<int:post_id>/like/', like_post, name='like_post'),
    path('post/<int:post_id>/comments/', get_comments, name='get_comments'),
    path('post/create/', create_post, name='create_post'),
    path('comment/<int:comment_id>/like/', like_comment, name='like_comment'),
    path('steamDB/api/search_games/', search_steam_games, name='search_games'),
    path('get_kline_data/', get_kline_data, name='get_kline_data'),
]

'http://127.0.0.1:8000/steamDB/api/by_tag?tag=a'
'http://127.0.0.1:8000/steamDB/api/tag_statistics'
'http://127.0.0.1:8000/steamDB/api/price_statistics'
'http://127.0.0.1:8000/steamDB/api/positive_rate_statistics'
