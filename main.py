import networkx as nx
import tkinter as tk
import tkinter.scrolledtext as scrolledText
from tkinter import messagebox
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Score import *
import gensim
from sentence_transformers import SentenceTransformer
import random

tk_window=tk.Tk()
tk_window.title("Metin Ozetleme")
tk_window.geometry("1300x750+110+25")

frame_label=tk.Frame(tk_window,padx=20,pady=20)
frame_label.pack()
frame_entry_and_select=tk.Frame(tk_window,padx=20,pady=0)
frame_entry_and_select.pack()
frame_graph=tk.Frame(tk_window,padx=20,pady=0)
frame_graph.pack(side=tk.LEFT, expand=1, padx=5,pady=50)
frame_summary=tk.Frame(tk_window,padx=20,pady=0)
frame_summary.pack(side=tk.RIGHT, expand=1, padx=5,pady=50)

graph = nx.Graph()

#word-embedding model yükleme
W2V_PATH="GoogleNews-vectors-negative300.bin"
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
#bert model yükleme
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

#nltk.download('averaged_perceptron_tagger')

def delete_graph_and_summary():

    for g in frame_graph.winfo_children():
        g.destroy()

    for g in frame_summary.winfo_children():
        g.destroy()


#butona basildiginda calisacak fonksiyon
def button_submit_click():

    global text_summary_bool,text_summary,lbl_rouge_score,text_srouge_score

    delete_graph_and_summary()


    if(document_path=="" or document_path2=="" or entry_sim_threshold.get()=="" or entry_score_threshold.get()==""):
        messagebox.showerror("Error","Tüm parametreleri giriniz")
        return "exit"


    try:
        float(entry_sim_threshold.get())
    except(Exception):
        messagebox.showerror("Error", "Cümle Benzerliği Thresholdu Parametresi Hatalı")
        return "exit"

    try:
        float(entry_score_threshold.get())
    except(Exception):
        messagebox.showerror("Error", "Cümle Skoru Thresholdu Parametresi Hatalı")
        return "exit"


    # dosya yolundan örnek metin okunuyor
    print(document_path)
    document = open(document_path, "r")

    """pass1 = ""
    while (pass1 == "" or pass1 == "\n"):
        pass1 = document.readline()"""

    title = ""
    while (title == "" or title == "\n"):
        title = document.readline()
    title = title.replace("\n", "")
    print(title)

    document_text = document.read()
    document_text = document_text.strip()
    document_text = document_text.replace("\n", "")

    #----------------------------------------------
    i=2
    while(i<len(document_text)-2):
        if document_text[i]==".":
            if (65<=ord(document_text[i+1]) and 90>=ord(document_text[i+1])) :
                document_text=document_text[0:(i+1)]+" "+document_text[(i+1):]
                i+=1
        i+=1

    #sentences = document_text.split(".")
    sentences = nltk.sent_tokenize(document_text)

    i = 0
    while (i < len(sentences)):
        if (sentences[i] == "" or len(sentences[i].split())==0):
            sentences.pop(i)
            i -= 1
        i += 1

    # ---------------------------------------------
    document.close()

    """document_content = document_content.split("Ã–rnek Ã–zet (Beklenen Ã‡Ä±ktÄ±):")
    input = document_content[0].strip()
    output = document_content[1].strip()

    exp_summary= output.split(".")
    if exp_summary[len(exp_summary) - 1] == "":
        exp_summary = exp_summary[0:len(exp_summary) - 1]"""

    #dosya yolundan beklenen özet okunuyor.
    print(document_path2)
    document2 = open(document_path2, "r")

    """pass1 = ""
    while (pass1 == "" or pass1 == "\n"):
        pass1 = document.readline()"""

    document_summary = document2.read()
    document_summary = document_summary.strip()
    document_summary = document_summary.replace("\n", "")

    #-----------------------------------------------------
    i = 2
    while (i < len(document_summary) - 2):
        if document_summary[i] == ".":
            if (65 <= ord(document_summary[i + 1]) and 90 >= ord(document_summary[i + 1])):
                document_summary = document_summary[0:(i + 1)] + " " + document_summary[(i + 1):]
                i += 1
        i += 1

    #exp_summary = document_summary.split(".")
    exp_summary= nltk.sent_tokenize(document_summary)

    i = 0
    while (i < len(exp_summary)):
        if (exp_summary[i] == "" or len(exp_summary[i].split()) == 0):
            exp_summary.pop(i)
            i -= 1
        i += 1
    # ---------------------------------------------

    document2.close()

    print(sentences)
    print("-------")
    print(exp_summary)



    ##secilen similarity threshold unu verir
    sim_threshold=float(entry_sim_threshold.get())
    print(sim_threshold)

    ## Bu score u gecen cumleler ozet icin secilecek
    score_threshold=float(entry_score_threshold.get())
    print(score_threshold)

    ## bu algoritma kullanilarak cumle benzerligi hesaplanacak
    print(val.get())

    # Cümleler üzerine hesaplamalar
    # dosya okuma olduktan sonra başlık ve cümle listesi elde edilecek
    #title="Gallery unveils interactive tree"

    # nltk
    nltk_sentences=[]
    for i in sentences:
        nltk_sentences.append(nltk_preprocessing(i))
    nltk_title=nltk_preprocessing(title)

    print(nltk_sentences)
    document_embeddings = sbert_model.encode(nltk_sentences)

    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(111)
    # plt.axis("off")

    score_list=[]
    ##i ve j döngüsünde 11 yerine (metindeki cümle sayısı+1) gelmeli
    for i in range(1, len(sentences)+1):

        if(i == 1):
            graph.add_node("Cümle " + str(i), sentences=sentences[0],param3=0)

        for j in range(i + 1, len(sentences)+1):

            if (i == 1):
                graph.add_node("Cümle " + str(j), sentences=sentences[j-1],param3=0)

            if(val.get()=="Word Embedding"):
                ##"Cümle " + str(i), "Cümle " + str(j) arasındaki similarity hesaplanır, similarity ye atanır (0.5 yerine) (word embedding ile)
                similarity = word_embedding(model_w2v,nltk_sentences[i-1],nltk_sentences[j-1])

            else:
                ##"Cümle " + str(i), "Cümle " + str(j) arasındaki similarity hesaplanır, similarity ye atanır (0.5 yerine) (BERT ile)
                similarity = round(spatial.distance.cosine(document_embeddings[i-1], document_embeddings[j-1]),2)

            graph.add_edge("Cümle " + str(i), "Cümle " + str(j), similarity=similarity)


            if (similarity>=sim_threshold):
                graph.nodes["Cümle " + str(i)]["param3"]+=1
                graph.nodes["Cümle " + str(j)]["param3"]+=1
                graph.add_edge("Cümle " + str(i), "Cümle " + str(j), color="lightgreen")
                graph.add_edge("Cümle " + str(i), "Cümle " + str(j), alpha=1)
            else:
                graph.add_edge("Cümle " + str(i), "Cümle " + str(j), color="white")
                graph.add_edge("Cümle " + str(i), "Cümle " + str(j), alpha=0)


        ## score hesaplanır, 0.0 yerine yazılır
        p3=(graph.nodes["Cümle " + str(i)]["param3"])/(len(sentences)*((len(sentences)-1)/2))
        top_words=get_top_words(nltk_sentences)

        score=(p1(sentences[i-1])*0.26216368)+ p2(sentences[i-1])*0.22589249+p3*1.05450876+p4(nltk_title,nltk_sentences[i-1])*0.86990408+p5(nltk_sentences[i-1],top_words)*1.11945169
        print(f"p1: {p1(sentences[i-1])} p2: {p2(sentences[i-1])} p3: {p3} p4: {p4(nltk_title,nltk_sentences[i-1])} p5: {p5(nltk_sentences[i-1],top_words)}")
        tuple=(i,score)
        score_list.append(tuple)
        graph.nodes["Cümle " + str(i)]["score"] = score


    list=normalization(score_list)
    for i in range(1,len(list)+1):
        (index,score)=list[i-1]
        #--------------------------
        graph.nodes["Cümle "+ str(index)]["score"]=round(score,2)
        #--------------------------

    sorted_tuple=sorted(list,key=lambda x:(-x[1],x[0]))
    summary=""
    for i in sorted_tuple:
        (index,score)=i
        if score>=score_threshold:
            summary=summary+sentences[index-1] + " "

    summary_list=summary.split('.')
    summary_list=summary_list[:-1]
    rouge_score=rouge(exp_summary,summary_list)
    print(rouge_score)
    rouge_s=rouge_score['rouge1']


    pos = nx.spring_layout(graph, scale=2)

    pos_node_attributes_score = {}
    for a, b in graph.nodes(data=True):
        pos_node_attributes_score[a] = (pos[a][0] - 0.1, pos[a][1] + 0.3)

    pos_node_attributes_param3 = {}
    for a, b in graph.nodes(data=True):
        pos_node_attributes_param3[a] = (pos[a][0] + 0.1, pos[a][1] + 0.3)


    edge_labels = {(a, b): (c["similarity"]) for a, b, c in graph.edges(data=True)}

    node_labels_score = {(a): (b["score"]) for a, b in graph.nodes(data=True)}
    node_labels_param3 = {(a): (b["param3"]) for a, b in graph.nodes(data=True)}

    nx.draw(graph, pos, with_labels=True, node_color="blue", node_size=800, font_color="white", font_size="7",
            font_family="Times New Roman", font_weight="bold", edge_color="gray", width=1, ax=ax)

    [nx.draw_networkx_edge_labels(graph, pos, edge_labels={b: edge_labels[b]}, label_pos=0.3, font_size="6",
                                  bbox=dict(fc=graph.edges[b]["color"], ec=graph.edges[b]["color"], boxstyle="square",
                                            lw=0, alpha=graph.edges[b]["alpha"]), ax=ax) for a, b in
     enumerate(graph.edges())]

    nx.draw_networkx_labels(graph, pos=pos_node_attributes_score, labels=node_labels_score, font_color="black",
                            font_size="8", font_weight="bold",
                            bbox=dict(fc="pink", ec="lightgreen", boxstyle="square", lw=0), ax=ax)
    nx.draw_networkx_labels(graph, pos=pos_node_attributes_param3, labels=node_labels_param3, font_color="black",
                            font_size="8", font_weight="bold",
                            bbox=dict(fc="orange", ec="lightgreen", boxstyle="square", lw=0), ax=ax)

    canvas = FigureCanvasTkAgg(f, master=frame_graph)
    canvas.get_tk_widget().pack()



    text_summary = scrolledText.ScrolledText(frame_summary, font=("Arial", 11), fg="black", bg="white", width=50, height=28,
                                                  wrap="word", padx=10, pady=10)
    text_summary_bool = True
    text_summary.pack(pady=5)

    ##özet, "aa" yerine eklenmeli
    text_summary.insert(1.0, summary)
    text_summary.config(state=tk.DISABLED)


    lbl_rouge_score = tk.Label(frame_summary, text="ROUGE Score", font=("Arial", 11), fg="black", padx=10, height=1)
    lbl_rouge_score.pack(pady=0)
    text_srouge_score = tk.Text(frame_summary, font=("Arial", 11), fg="black", padx=2, height=1, width=20)
    text_srouge_score.pack(pady=1)

    #ROUGE score, "0.0" yerine eklenmeli
    text_srouge_score.tag_configure("center", justify='center')
    text_srouge_score.insert(1.0, round(rouge_s,2))
    text_srouge_score.tag_add("center", 1.0, "end")
    text_srouge_score.config(state=tk.DISABLED)

def load_document(value):
    global val1,document_path,select_load_document

    document_path=filedialog.askopenfilename(title="Doküman Yükle")

    if document_path.endswith("txt")==True:
        select_load_document.destroy()

        file_name=document_path.split("/")
        file_name=file_name[len(file_name)-1]

        val1 = tk.StringVar(value=file_name)
        list = ["Yeni Doküman Yükle"]

        select_load_document = tk.OptionMenu(frame_entry_and_select, val1, *list, command=load_document)
        select_load_document.config(width=17, bg="white", font=("Arial 10"))
        select_load_document.grid(row=1, column=0, padx=35, pady=0)

    else:
        select_load_document.destroy()

        val1 = tk.StringVar(value="")
        select_load_document = tk.OptionMenu(frame_entry_and_select, val1, "Doküman Yükle", command=load_document)
        select_load_document.config(width=17, bg="white", font=("Arial 10"))
        select_load_document.grid(row=1, column=0, padx=35, pady=0)

        document_path=""


def load_document2(value):
    global val2,document_path2,select_load_document2

    document_path2=filedialog.askopenfilename(title="Doküman Yükle")

    if document_path2.endswith("txt")==True:
        select_load_document2.destroy()

        file_name=document_path2.split("/")
        file_name=file_name[len(file_name)-1]

        val2 = tk.StringVar(value=file_name)
        list = ["Yeni Doküman Yükle"]

        select_load_document2 = tk.OptionMenu(frame_entry_and_select, val2, *list, command=load_document2)
        select_load_document2.config(width=17, bg="white", font=("Arial 10"))
        select_load_document2.grid(row=1, column=1, padx=30, pady=0)

    else:
        select_load_document2.destroy()

        val2 = tk.StringVar(value="")
        select_load_document2 = tk.OptionMenu(frame_entry_and_select, val2, "Doküman Yükle", command=load_document2)
        select_load_document2.config(width=17, bg="white", font=("Arial 10"))
        select_load_document2.grid(row=1, column=1, padx=30, pady=0)

        document_path2=""



lbl_load_document=tk.Label(frame_label,text="Metin Dokümanı Yükleme",font=("Arial",10),fg="black",padx=10,height=1)
lbl_load_document.grid(row=0,column=0,padx=30,pady=0)

lbl_load_document_sum=tk.Label(frame_label,text="Özet Dokümanı Yükleme",font=("Arial",10),fg="black",padx=10,height=1)
lbl_load_document_sum.grid(row=0,column=1,padx=30,pady=0)

lbl_sim_threshold=tk.Label(frame_label,text="Cümle Benzerliği Thresholdu",font=("Arial",10),fg="black",padx=10,height=1)
lbl_sim_threshold.grid(row=0,column=2,padx=30,pady=0)
lbl_score_threshold=tk.Label(frame_label,text="Cümle Skoru Thresholdu",font=("Arial",10),fg="black",padx=10,height=1)
lbl_score_threshold.grid(row=0,column=3,padx=0,pady=0)
lbl_algorithm=tk.Label(frame_label,text="Cümle Benzerliği Algoritması",font=("Arial",10),fg="black",padx=10,height=1)
lbl_algorithm.grid(row=0,column=4,padx=30,pady=0)
lbl_empty=tk.Label(frame_label,text="                  ",font=("Arial",11),fg="black",padx=10,height=1)
lbl_empty.grid(row=0,column=5,padx=30,pady=0)




document_path=""
val1=tk.StringVar(value="")
list=["Doküman Yükle"]
select_load_document=tk.OptionMenu(frame_entry_and_select,val1,*list,command=load_document)
select_load_document.config(width=17,bg="white",font=("Arial 10"))
select_load_document.grid(row=1,column=0,padx=35,pady=0)

document_path2=""
val2=tk.StringVar(value="")
select_load_document2=tk.OptionMenu(frame_entry_and_select,val2,*list,command=load_document2)
select_load_document2.config(width=17,bg="white",font=("Arial 10"))
select_load_document2.grid(row=1,column=1,padx=30,pady=0)


entry_sim_threshold=tk.Entry(frame_entry_and_select,font=("Arial 10"),fg="black",justify="center")
entry_sim_threshold.grid(row=1,column=2,padx=55,pady=0)
entry_score_threshold=tk.Entry(frame_entry_and_select,font=("Arial 10"),fg="black",justify="center")
entry_score_threshold.grid(row=1,column=3,padx=12,pady=0)
val=tk.StringVar(value="Word Embedding")
algorithm_list={"Word Embedding","BERT"}
select_algorithm=tk.OptionMenu(frame_entry_and_select,val,"Word Embedding","BERT")
select_algorithm.config(width=17,bg="white",font=("Arial 10"))
select_algorithm.grid(row=1,column=4,padx=43,pady=0)
button_submit=tk.Button(frame_entry_and_select,text="SHOW GRAPH",font=("Arial 10"),bg="blue",fg="white",cursor="plus",command=button_submit_click)
button_submit.grid(row=1,column=5,padx=20,pady=0)




"""text_summary=scrolledText.ScrolledText(frame_summary,font=("Arial",11),fg="black",bg="white",width=50,height=30,wrap="word",padx=10,pady=10)
lbl_rouge_score = tk.Label(frame_summary, text="ROUGE Score", font=("Arial", 10), fg="black", padx=10, height=1)
text_srouge_score = tk.Text(frame_summary, font=("Arial", 10), fg="black", padx=2, height=1,width=20)"""



tk_window.mainloop()