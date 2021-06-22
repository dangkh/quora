from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import torch
import torch.nn as nn

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 
                       "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
                       "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", 
                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                       "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" ,
                      "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


puncts1 = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~"
puncts2 = '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
puncts3 = list(puncts1) + list(puncts2) + [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

BYTE_REPLACE = dict([(u, f' {u} ') for u in puncts3])

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def clean_special_chars(text):
   
    for p in puncts3:
        if p in text:
            text = text.replace(p, f' {p} ')        
    #text = text.translate(BYTE_REPLACE)   
    
    return text

def correct_spelling(x):
    for word in mispell_dict.keys():
        if word in x:
            x = x.replace(word, mispell_dict[word])
        
    specials = {'\u200b': ' ', '…': ' ', '\ufeff': '', 'करना': '', 'है': '', u'\xa0':' ', '\n':' '}  
    for s in specials:
        x = x.replace(s, specials[s])
    
    return x

def clean_contractions(text):    
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    return text

def preprocess(text):
    text = clean_contractions(text)
    text = correct_spelling(text)
    text = clean_special_chars(text)
    text = added_clean_misspell(text)
    
    return text


misspell_mapping = {'Terroristan': 'terrorist Pakistan', 'terroristan': 'terrorist Pakistan',
                    'FATF': 'Western summit conference',
                    'BIMARU': 'BIMARU Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh', 'Hinduphobic': 'Hindu phobic',
                    'hinduphobic': 'Hindu phobic', 'Hinduphobia': 'Hindu phobic', 'hinduphobia': 'Hindu phobic',
                    'Babchenko': 'Arkady Arkadyevich Babchenko faked death', 'Boshniaks': 'Bosniaks',
                    'Dravidanadu': 'Dravida Nadu', 'mysoginists': 'misogynists', 'MGTOWS': 'Men Going Their Own Way',
                    'mongloid': 'Mongoloid', 'unsincere': 'insincere', 'meninism': 'male feminism',
                    'jewplicate': 'jewish replicate', 'jewplicates': 'jewish replicate', 'andhbhakts': 'and Bhakt',
                    'unoin': 'Union', 'daesh': 'Islamic State of Iraq and the Levant', 'burnol': 'movement about Modi',
                    'Kalergi': 'Coudenhove-Kalergi', 'Bhakts': 'Bhakt', 'bhakts': 'Bhakt', 'Tambrahms': 'Tamil Brahmin',
                    'Pahul': 'Amrit Sanskar', 'SJW': 'social justice warrior', 'SJWs': 'social justice warrior',
                    ' incel': ' involuntary celibates', ' incels': ' involuntary celibates', 'emiratis': 'Emiratis',
                    'weatern': 'western', 'westernise': 'westernize', 'Pizzagate': 'debunked conspiracy theory',
                    'naïve': 'naive', 'Skripal': 'Russian military officer', 'Skripals': 'Russian military officer',
                    'Remainers': 'British remainer', 'Novichok': 'Soviet Union agents',
                    'gauri lankesh': 'Famous Indian Journalist', 'Castroists': 'Castro supporters',
                    'remainers': 'British remainer', 'bremainer': 'British remainer', 'antibrahmin': 'anti Brahminism',
                    'HYPSM': ' Harvard, Yale, Princeton, Stanford, MIT', 'HYPS': ' Harvard, Yale, Princeton, Stanford',
                    'kompromat': 'compromising material', 'Tharki': 'pervert', 'tharki': 'pervert',
                    'mastuburate': 'masturbate', 'Zoë': 'Zoe', 'indans': 'Indian', ' xender': ' gender',
                    'Naxali ': 'Naxalite ', 'Naxalities': 'Naxalites', 'Bathla': 'Namit Bathla',
                    'Mewani': 'Indian politician Jignesh Mevani', 'Wjy': 'Why',
                    'Fadnavis': 'Indian politician Devendra Fadnavis', 'Awadesh': 'Indian engineer Awdhesh Singh',
                    'Awdhesh': 'Indian engineer Awdhesh Singh', 'Khalistanis': 'Sikh separatist movement',
                    'madheshi': 'Madheshi', 'BNBR': 'Be Nice, Be Respectful',
                    'Jair Bolsonaro': 'Brazilian President politician', 'XXXTentacion': 'Tentacion',
                    'Slavoj Zizek': 'Slovenian philosopher',
                    'borderliners': 'borderlines', 'Brexit': 'British Exit', 'Brexiter': 'British Exit supporter',
                    'Brexiters': 'British Exit supporters', 'Brexiteer': 'British Exit supporter',
                    'Brexiteers': 'British Exit supporters', 'Brexiting': 'British Exit',
                    'Brexitosis': 'British Exit disorder', 'brexit': 'British Exit',
                    'brexiters': 'British Exit supporters', 'jallikattu': 'Jallikattu', 'fortnite': 'Fortnite',
                    'Swachh': 'Swachh Bharat mission campaign ', 'Quorans': 'Quora users', 'Qoura': 'Quora',
                    'quoras': 'Quora', 'Quroa': 'Quora', 'QUORA': 'Quora', 'Stupead': 'stupid',
                    'narcissit': 'narcissist', 'trigger nometry': 'trigonometry',
                    'trigglypuff': 'student Criticism of Conservatives', 'peoplelook': 'people look',
                    'paedophelia': 'paedophilia', 'Uogi': 'Yogi', 'adityanath': 'Adityanath',
                    'Yogi Adityanath': 'Indian monk and Hindu nationalist politician',
                    'Awdhesh Singh': 'Commissioner of India', 'Doklam': 'Tibet', 'Drumpf ': 'Donald Trump fool ',
                    'Drumpfs': 'Donald Trump fools', 'Strzok': 'Hillary Clinton scandal', 'rohingya': 'Rohingya ',
                    ' wumao ': ' cheap Chinese stuff ', 'wumaos': 'cheap Chinese stuff', 'Sanghis': 'Sanghi',
                    'Tamilans': 'Tamils', 'biharis': 'Biharis', 'Rejuvalex': 'hair growth formula Medicine',
                    'Fekuchand': 'PM Narendra Modi in India', 'feku': 'Feku', 'Chaiwala': 'tea seller in India',
                    'Feku': 'PM Narendra Modi in India ', 'deplorables': 'deplorable', 'muhajirs': 'Muslim immigrant',
                    'Gujratis': 'Gujarati', 'Chutiya': 'Tibet people ', 'Chutiyas': 'Tibet people ',
                    'thighing': 'masterbate between the legs of a female infant', '卐': 'Nazi Germany',
                    'Pribumi': 'Native Indonesian', 'Gurmehar': 'Gurmehar Kaur Indian student activist',
                    'Khazari': 'Khazars', 'Demonetization': 'demonetization', 'demonetisation': 'demonetization',
                    'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                    'antinationals': 'antinational', 'Cryptocurrencies': 'cryptocurrency',
                    'cryptocurrencies': 'cryptocurrency', 'Hindians': 'North Indian', 'Hindian': 'North Indian',
                    'vaxxer': 'vocal nationalist ', 'remoaner': 'remainer ', 'bremoaner': 'British remainer ',
                    'Jewism': 'Judaism', 'Eroupian': 'European', "J & K Dy CM H ' ble Kavinderji": '',
                    'WMAF': 'White male married Asian female', 'AMWF': 'Asian male married White female',
                    'moeslim': 'Muslim', 'cishet': 'cisgender and heterosexual person', 'Eurocentrics': 'Eurocentrism',
                    'Eurocentric': 'Eurocentrism', 'Afrocentrics': 'Africa centrism', 'Afrocentric': 'Africa centrism',
                    'Jewdar': 'Jew dar', 'marathis': 'Marathi', 'Gynophobic': 'Gyno phobic',
                    'Trumpanzees': 'Trump chimpanzee fool', 'Crimean': 'Crimea people ', 'atrracted': 'attract',
                    'Myeshia': 'widow of Green Beret killed in Niger', 'demcoratic': 'Democratic', 'raaping': 'raping',
                    'feminazism': 'feminism nazi', 'langague': 'language', 'sathyaraj': 'actor',
                    'Hongkongese': 'HongKong people', 'hongkongese': 'HongKong people', 'Kashmirians': 'Kashmirian',
                    'Chodu': 'fucker', 'penish': 'penis',
                    'chitpavan konkanastha': 'Hindu Maharashtrian Brahmin community',
                    'Madridiots': 'Real Madrid idiot supporters', 'Ambedkarite': 'Dalit Buddhist movement ',
                    'ReleaseTheMemo': 'cry for the right and Trump supporters', 'harrase': 'harass',
                    'Barracoon': 'Black slave', 'Castrater': 'castration', 'castrater': 'castration',
                    'Rapistan': 'Pakistan rapist', 'rapistan': 'Pakistan rapist', 'Turkified': 'Turkification',
                    'turkified': 'Turkification', 'Dumbassistan': 'dumb ass Pakistan', 'facetards': 'Facebook retards',
                    'rapefugees': 'rapist refugee', 'Khortha': 'language in the Indian state of Jharkhand',
                    'Magahi': 'language in the northeastern Indian', 'Bajjika': 'language spoken in eastern India',
                    'superficious': 'superficial', 'Sense8': 'American science fiction drama web television series',
                    'Saipul Jamil': 'Indonesia artist', 'bhakht': 'bhakti', 'Smartia': 'dumb nation',
                    'absorve': 'absolve', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Whta': 'What',
                    'esspecial': 'especial', 'doI': 'do I', 'theBest': 'the best',
                    'howdoes': 'how does', 'Etherium': 'Ethereum', '2k17': '2017', '2k18': '2018', 'qiblas': 'qibla',
                    'Hello4 2 cab': 'Online Cab Booking', 'bodyshame': 'body shaming', 'bodyshoppers': 'body shopping',
                    'bodycams': 'body cams', 'Cananybody': 'Can any body', 'deadbody': 'dead body',
                    'deaddict': 'de addict', 'Northindian': 'North Indian ', 'northindian': 'north Indian ',
                    'northkorea': 'North Korea', 'koreaboo': 'Korea boo ',
                    'Brexshit': 'British Exit bullshit', 'shitpost': 'shit post', 'shitslam': 'shit Islam',
                    'shitlords': 'shit lords', 'Fck': 'Fuck', 'Clickbait': 'click bait ', 'clickbait': 'click bait ',
                    'mailbait': 'mail bait', 'healhtcare': 'healthcare', 'trollbots': 'troll bots',
                    'trollled': 'trolled', 'trollimg': 'trolling', 'cybertrolling': 'cyber trolling',
                    'sickular': 'India sick secular ', 'Idiotism': 'idiotism',
                    'Niggerism': 'Nigger', 'Niggeriah': 'Nigger'}

def added_clean_misspell(text):
    for bad_word in misspell_mapping:
        if bad_word in text:
            text = text.replace(bad_word, misspell_mapping[bad_word])
    return text

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
        hidden_size = 60
        self.hidden_size = 60
        self.embedding = nn.Embedding(250000, 300)
        
        self.embedding_dropout = nn.Dropout(0.25)
        self.lstm = nn.LSTM(300, hidden_size, bidirectional=True, batch_first=True)        
        
        output_layer = 2*hidden_size
        self.linear = nn.Linear(output_layer, output_layer)
        self.bn_linear = nn.BatchNorm1d(output_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Linear(output_layer, 1)
    
    def forward(self, x):
        h_embedding = self.embedding(x)
        
        h_lstm, _ = self.lstm(h_embedding)
        
        # global max pooling
        avg_pool = torch.mean(h_lstm, 1)
        
        conc = self.relu(self.bn_linear(self.linear(avg_pool)))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        return out

# s = "how could black people dominate the world?"

# s = preprocess(s)
# with open('models/tokenizer.pickle', 'rb') as handle:
#   newTokenizer = pickle.load(handle)
# ss = newTokenizer.texts_to_sequences([s])
# print(ss)
# abc = pad_sequences(ss, maxlen=70000)
# print(abc)
# train_X = pad_sequences(train_X, maxlen=maxlen)
# test_X = pad_sequences(test_X, maxlen=maxlen)