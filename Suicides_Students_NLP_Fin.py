#Web Scraping of Suicidal News by Students, for the last 2 years due to Academic Pressure and Distress!
#Natural Language Processing - NLP.

#Installing General utility libraries..
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sn

#Web Scraping related libraries..
from bs4 import BeautifulSoup as bs #Web scraper
import urllib.request
import requests #Request to URL source link
import re #Regular Expression

#NLP Related Libraries/Packages..
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller

#Dedicated -- Newspaper/Article Extraction Library..
pip install newspaper3k
from newspaper import Article

##############################################

#Web scraping from different links to collate the information(news content) together. 
#25 Links

urllinks = ['https://www.indiatoday.in/cities/hyderabad/story/iit-hyderbad-student-dies-by-suicide-due-to-stress-of-not-getting-job-suicide-note-found-2000025-2022-09-14',
            'https://www.newindianexpress.com/cities/vijayawada/2022/feb/07/depressed-by-academic-pressure-intermediate-student-commits-suicide-2416310.html',
            'https://www.edexlive.com/news/2021/dec/23/yet-another-student-in-telangana-dies-by-suicide-after-inter-exam-results-death-toll-rises-to-6-26451.html',
            'https://timesofindia.indiatimes.com/city/ahmedabad/gujarat-btech-student-caught-in-cheating-case-dies-by-suicide/articleshow/91524033.cms',
            'https://timesofindia.indiatimes.com/city/visakhapatnam/andhra-pradesh-iiit-student-dies-by-suicide-in-srikakulam-hostel-room/articleshow/94065121.cms',
            'https://timesofindia.indiatimes.com/city/delhi/mumbai-girl-dies-by-suicide-after-parents-scold-her-about-studies/articleshow/94423946.cms',
            'https://indianexpress.com/article/cities/chennai/another-neet-aspirant-suicide-tamil-nadu-7507958/',
            'https://www.indiatoday.in/india/story/tamil-nadu-student-suicide-4th-case-1980370-2022-07-27',
            'https://timesofindia.indiatimes.com/city/indore/college-student-attempts-suicide-alleging-ragging/articleshow/92760456.cms',
            'https://timesofindia.indiatimes.com/city/chandigarh/principal-booked-for-students-suicide/articleshow/93628148.cms',
            'https://timesofindia.indiatimes.com/city/nagpur/nagpur-student-jumps-to-death-from-college-building/articleshow/93532179.cms',
            'https://timesofindia.indiatimes.com/city/lucknow/nursing-student-ends-life/articleshow/93199969.cms',
            'https://timesofindia.indiatimes.com/city/mangaluru/bba-student-dies-by-suicide/articleshow/92288612.cms',
            'https://timesofindia.indiatimes.com/city/bengaluru/bengaluru-bsc-student-writes-torture-in-note-ends-life/articleshow/92760127.cms',
            'https://timesofindia.indiatimes.com/city/kochi/boy-found-dead-at-house-suicide-suspected/articleshow/94467335.cms',
            'https://timesofindia.indiatimes.com/city/lucknow/up-unable-to-meet-delhi-university-cutoff-girl-from-sitapur-jumps-into-river-dies/articleshow/92374681.cms',
            'https://timesofindia.indiatimes.com/city/thiruvananthapuram/girl-student-commits-suicide/articleshow/91730203.cms',
            'https://timesofindia.indiatimes.com/city/chennai/college-student-dies-by-suicide/articleshow/91627739.cms',
            'https://timesofindia.indiatimes.com/city/raipur/engg-student-dies-of-suicide-in-korba-district/articleshow/92447293.cms',
            'https://timesofindia.indiatimes.com/city/bhubaneswar/bjb-college-girl-student-dies-by-suicide-ragging-suspected/articleshow/92625113.cms',
            'https://timesofindia.indiatimes.com/city/bhubaneswar/after-ruchika-mohanty-another-bhubaneswar-college-student-dies-by-suicide/articleshow/92683781.cms',
            'https://timesofindia.indiatimes.com/city/rajkot/asked-to-quit-studies-college-girl-attempts-suicide/articleshow/94315160.cms',
            'https://timesofindia.indiatimes.com/city/bengaluru/scolded-by-kin-pu-student-ends-life/articleshow/91605292.cms',
            'https://timesofindia.indiatimes.com/city/chennai/iit-madras-student-hangs-himself/articleshow/94233487.cms',
            'https://timesofindia.indiatimes.com/city/indore/hounded-by-youth-to-meet-college-girl-dies-by-suicide/articleshow/93719889.cms']
            
urllinks

###################################################################

#Web Scraping the news content from various news articles/media pages/etc. via "from newspaper import Article" library
columns = [] #Open container
#For looping, for to loop every content/article extracted from sources
#Try-Exception block to catch Errors, if occurs
for url in urllinks:
    try:
        content = Article(url,language = 'en')
        content.download()
        content.parse()
        text = content.text
        print(text)
        cols = {'url':url,'text':text}
        columns.append(cols)
    except Exception as e:
        print(e)
        cols = {'url':url,'text':'NotApplicable'}
        columns.append(cols)

#To Data Frame format..
df_result = pd.DataFrame(columns) 
df_result

#Converting the Data Frame into List format..
text = df_result['text'].values.tolist()
text
corpus_text = ' '.join("".join(i) for i in text)
corpus_text 

#Writing and Storing into an external file..
with open("suicidalnews_New.txt","w",encoding = 'utf8') as op:
    op.write(str(corpus_text))
    

#Combine all the news content to one single corpus/paragraph..
join_newscon = "".join(corpus_text)
join_newscon

#Importing 'NLTK' main Package..
import nltk

#Removing unwanted characters / special symbols / numbers / other characters..
join_newscon = re.sub("[^A-Za-z" "]+"," ",join_newscon).lower()
join_newscon
join_newscon = re.sub("[0-9" "]+"," ",join_newscon)
join_newscon

#Splitting up the "words"..
join_newscon_words = join_newscon.split(" ")
join_newscon_words

#Implementing "TFIDF" technique from "sklearn" library
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(join_newscon_words,encoding = 'utf-8',use_idf=True,ngram_range=(1,3))
X = tvec.fit_transform(join_newscon_words)
X

#Importing 'Stopwords'..
with open('D:/360DigiTMG__DS_AI/Data Science/Project Docs/PROJECT 2/stopwords_en.txt','r') as ri:
    stpwrds = ri.read()
stpwrds = stpwrds.split("\n")
stpwrds.extend(['institute','student','purportedly','suicide','district','terrace','CrPC','OneLife','READ ALSO','helpline','Also Read','vandalised','Number','advertisement',
                'police','students','room','case','year','college','girl','note','hospital','allegedly','number','step','day','body','committed','death','due','found','members','class',
                'hostel','family','investigation','told','reason','station','registered','reportedly','resident'])
stpwrds

#Looping in the words,by neglecting stop words..
join_newscon_words = [w for w in join_newscon_words if not w in stpwrds]
join_newscon_words

#Joining the news content to a large paragraph corpus textual data..
join_newscon = " ".join(join_newscon_words)
join_newscon

#Generating the General Word Cloud..
pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

gen_wcld = WordCloud(background_color='white',width = 500,height=500,margin=2,prefer_horizontal=0.8,max_words=5000).generate(join_newscon)
gen_wcld
plt.figure(1)
plt.title("General WordCloud")
plt.imshow(gen_wcld)
plt.axis("off")
plt.show()

########################################

#POSITIVE -- Word Cloud..
with open('D:/360DigiTMG__DS_AI/Data Science/Project Docs/PROJECT 2/positive-words.txt','r') as pi:
    pwords = pi.read().split("\n")
    pwords
    
#Joining with Looping..
join_newscon_pos_words = " ".join([w for w in join_newscon_words if w in pwords])
join_newscon_pos_words

gen_wcld_pos = WordCloud(background_color='black',width = 500,height=500,margin=3,prefer_horizontal=0.8,max_words=5000).generate(join_newscon_pos_words)
gen_wcld_pos
plt.figure(2)
plt.title("Positive WordCloud")
plt.imshow(gen_wcld_pos)
plt.axis("off")
plt.show()

#NEGATIVE -- Word Cloud..
with open('D:/360DigiTMG__DS_AI/Data Science/Project Docs/PROJECT 2/negative-words.txt','r') as ni:
    nwords = ni.read().split("\n")
    nwords
    
#Joining with Looping..
join_newscon_neg_words = " ".join([w for w in join_newscon_words if w in nwords])
join_newscon_neg_words

gen_wcld_neg = WordCloud(background_color='black',width = 500,height=500,margin=3,prefer_horizontal=0.8,max_words=5000).generate(join_newscon_neg_words)
gen_wcld_neg
plt.figure(3)
plt.title("Negative WordCloud")
plt.imshow(gen_wcld_neg)
plt.axis("off")
plt.show()

#Tokenization and Lemmatization steps..
import nltk
nltk.download('punkt')
lemma = nltk.WordNetLemmatizer()
lemma

#Converting to Lower Case, replace and tokenizing..
txt = join_newscon.lower()
txt
txt = txt.replace("'","")
txt
tokens = nltk.word_tokenize(txt)
tokens
txt1 = nltk.Text(tokens)
txt1

#Removing Extra Special Characters and Stop words..
txt_clean = [' '.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]",word)) for word in txt1]
txt_clean

#Defining the Default library 'STOPWORDS'..
def_set_stpwrds = set(STOPWORDS)
def_set_stpwrds

#Removing further more extra stop words in corpus textual data which does not give/add any meaning to it..
custm_words = ['place','gatekeeper','likely','Cost','comment','reported','government','intermediate','extreme','home','school','died',
               'state','total','education','including','risk','news','impact','access','covid','cent']
custm_words
new_stop_words = def_set_stpwrds.union(custm_words)
new_stop_words 

#Removing stop words once again!
txt_clean = [w for w in txt_clean if w not in new_stop_words]
txt_clean

#Taking the non-empty entries/news content data only!
txt_clean = [s for s in txt_clean if len(s)!=0]
txt_clean

#Package Library
#Lemmatization technique
nltk.download('wordnet')
txt_clean = [lemma.lemmatize(t) for t in txt_clean]
txt_clean

#Tokenizing the words..
nltk_tok = nltk.word_tokenize(txt)
nltk_tok
#tokens = nltk.word_tokenize(txt)
#tokens
#txt1 = nltk.Text(tokens)
#txt1

#BIGRAM Pattern Representation!
bigram_word = list(nltk.bigrams(txt_clean))
bigram_word
print("The following words are BIGRAM Pattern")
print(bigram_word)

bigram_words = [' '.join(word) for word in bigram_word]
bigram_words
print("The following words are BIGRAM words Representation")
print(bigram_words)

#To find the Frequency of BIGRAMS using TFIDF Count Vectorizer..
from sklearn.feature_extraction.text import CountVectorizer
bi_vec = CountVectorizer(ngram_range = (2,2))
bigram_freq = bi_vec.fit_transform(bigram_words)
bigram_freq
#Number of words in BIGRAM..
bi_vec.vocabulary_ #BIGRAM Words Representation

sum_words = bigram_freq.sum(axis=0) #axis = 1 
words_freq = [(word, sum_words[0, idx]) for word, idx in bi_vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True) #"Lambda" --> nameless function in descending order..& in Descending orderwise
print(words_freq[:100])

#Refined WordCloud Representation
words_dict = dict(words_freq)
words_dict
fin_wcld = WordCloud(background_color='white',width = 500, height = 500,margin = 4,prefer_horizontal=0.8,max_words=5000).generate_from_frequencies(words_dict)
fin_wcld
plt.figure(4)
plt.title("Frequently occuring BIGRAM Patternized words with same nature")
plt.imshow(fin_wcld,interpolation = 'bilinear')
plt.axis("off")
plt.show()

####################################################################################

import nltk
from nltk.util import trigrams

#TRIGRAM Pattern Representation!
trigram_word = list(nltk.trigrams(txt_clean))
trigram_word
print("The following words are TRIGRAM Pattern")
print(trigram_word)

trigram_words = [' '.join(word) for word in trigram_word]
trigram_words
print("The following words are TRIGRAM words Representation")
print(trigram_words)

#To find the Frequency of TRIGRAMS using TFIDF Count Vectorizer..
from sklearn.feature_extraction.text import CountVectorizer
tri_vec = CountVectorizer(ngram_range = (3,3))
trigram_freq = tri_vec.fit_transform(trigram_words)
trigram_freq
#Number of words in TRIGRAM..
tri_vec.vocabulary_ #TRIGRAM Words Representation

sum_words = trigram_freq.sum(axis=0) #axis = 1 
words_freq = [(word, sum_words[0, idx]) for word, idx in tri_vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True) #"Lambda" --> nameless function in descending order..& in Descending orderwise
print(words_freq[:100])

#Refined WordCloud Representation
words_dict = dict(words_freq)
words_dict
fin_wcld_tri = WordCloud(background_color='red',width = 500, height = 500,margin = 5,prefer_horizontal=0.8,max_words=5000).generate_from_frequencies(words_dict)
fin_wcld_tri
plt.figure(5)
plt.title("Frequently occuring TRIGRAM Patternized words with same nature")
plt.imshow(fin_wcld_tri,interpolation = 'bilinear')
plt.axis("off")
plt.show()

#######################################################################################