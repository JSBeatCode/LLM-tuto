# 02_01_llm_orderCoding.py 의 결과를 그대로 복붙해서 실행함.
import requests

def get_news(query):
    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=30&sort=sim"
    headers = {
        'X-Naver-Client-Id': 'NFIs2bJuLpFyUJJ1GA1o',
        'X-Naver-Client-Secret': 'c6qysGTr27'
    }
    
    response = requests.get(url, headers=headers)
    news_items = response.json().get('items', [])
    
    result = ""
    for item in news_items:
        result += f"제목: {item['title']}\n"
        result += f"URL: {item['link']}\n"
        result += f"내용: {item['description']}\n"
        result += "---\n"
    
    return result.strip()

# 검색어
query = '생성형 AI'

# 검색 결과를 return하는 함수
result = get_news(query)
print(result)