# myapp/middleware.py

from django.shortcuts import redirect

class SessionCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 检查会话数据
        if 'user_id' not in request.session and request.path not in ['/steamDB/login/', '/steamDB/api/login/', '/steamDB/api/register/']:
            return redirect('login')
        response = self.get_response(request)
        return response