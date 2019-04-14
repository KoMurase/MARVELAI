from flickrapi import FlickrAPI
from urllib.request import urlretrieve #atomの補完機能
from pprint import pprint #途中のデータをプリント
import os,time,sys #osの情報を得るために

#APIキーの情報

key="fcbc826eeeba5871d78c0da7bbc956b9"
secret="81045080dd1bce62"

wait_time=1
#serverをパンクさせないためやスパムと勘違いされないために
#わざと間隔をあけている

#保存フォルダの指定
charactername = sys.argv[1]
savedir = "./" + charactername

flickrapi = FlickrAPI(key,secret,format='parsed-json')
result = flickrapi.photos.search(
    text=charactername,
    per_page=400,
    media='photos',
    sort='relevance',
    safe_search=1,
    extras='url_q,licence'
)

photos=result['photos']
#pprint(photos)

for i ,photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + './' + photo['id'] + '.jpg'
    if os.path.exists(filepath):continue
    #fileが存在しなければ↓
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)#アクセスが連発しないために
