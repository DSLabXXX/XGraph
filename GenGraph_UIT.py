__author__ = 'iankuoli'

#
# generate UIT graph
#
#  user \t item \t tag \t <UI> \t <UT> \t <IT>
#
#  <UI> = f_UI_1,f_UI_2,...
#

import networkx as nx
import math

dict_Uid2Uindex = dict()
dict_Iid2Iindex = dict()
dict_Tid2Tindex = dict()

valCount_U = 0
valCount_I = 0
valCount_T = 0

G_UI = nx.DiGraph()
G_UT = nx.DiGraph()
G_IT = nx.DiGraph()

f_UIT = open('lastfm/user_taggedartists.dat', 'r')
f_UI = open('lastfm/user_artists.dat', 'r')

for line in f_UI:
    try:
        l = line.strip('\n').split('\t')
        userID = int(l[0])
        itemID = int(l[1])
        weight = float(l[2])
    except:
        continue

    if userID in dict_Uid2Uindex.keys():
        userIndex = dict_Uid2Uindex[userID]
    else:
        userIndex = valCount_U
        dict_Uid2Uindex[userID] = userIndex
        valCount_U += 1

    if itemID in dict_Iid2Iindex.keys():
        itemIndex = dict_Iid2Iindex[itemID]
    else:
        itemIndex = valCount_I
        dict_Iid2Iindex[itemID] = itemIndex
        valCount_I += 1

    u = 'u' + str(userIndex)
    i = 'i' + str(itemIndex)
    G_UI.add_edge(u, i, w=weight)

for line in f_UIT:
    try:
        l = line.strip('\n').split('\t')
        userID = int(l[0])
        itemID = int(l[1])
        tagID = int(l[2])
    except:
        continue

    userIndex = -1
    itemIndex = -1
    tagIndex = -1

    if userID in dict_Uid2Uindex.keys():
        userIndex = dict_Uid2Uindex[userID]
    else:
        continue

    if itemID in dict_Iid2Iindex.keys():
        itemIndex = dict_Iid2Iindex[itemID]
    else:
        continue

    if tagID in dict_Tid2Tindex.keys():
        tagIndex = dict_Tid2Tindex[tagID]
    else:
        tagIndex = valCount_T
        dict_Tid2Tindex[tagID] = tagIndex
        valCount_T += 1

    u = 'u' + str(userIndex)
    i = 'i' + str(itemIndex)
    t = 't' + str(tagIndex)

    if G_UT.has_edge(u, t):
        G_UT[u][t]['w'] += 1
    else:
        G_UT.add_edge(u, t, w=1)

    if G_IT.has_edge(i, t):
        G_IT[i][t]['w'] += 1
    else:
        G_IT.add_edge(i, t, w=1)

f_Uid2Uindex = open('lastfm/map_Uid2Uindex.txt', 'w')
f_Iid2Iindex = open('lastfm/map_Iid2Iindex.txt', 'w')
f_Tid2Tindex = open('lastfm/map_Tid2Tindex.txt', 'w')

for uid in dict_Uid2Uindex.keys():
    f_Uid2Uindex.write(str(uid) + '\t' + str(dict_Uid2Uindex[uid]) + '\n')

for iid in dict_Iid2Iindex.keys():
    f_Iid2Iindex.write(str(iid) + '\t' + str(dict_Iid2Iindex[iid]) + '\n')

for tid in dict_Tid2Tindex.keys():
    f_Tid2Tindex.write(str(tid) + '\t' + str(dict_Tid2Tindex[tid]) + '\n')

f_UITGraph = open('lastfm/UIT_Graph.txt', 'w')

for edge_UI in G_UI.edges():
    u = edge_UI[0]
    i = edge_UI[1]
    w_UI = G_UI[u][i]['w']

    if u not in G_UT.nodes() or i not in G_IT.nodes():
        continue

    for t in G_UT.successors(u):

        if t not in G_IT.successors(i):
            continue

        w_UT = G_UT[u][t]['w']
        w_IT = G_IT[i][t]['w']

        uIdx = u[1:]
        iIdx = i[1:]
        tIdx = t[1:]

        f_UITGraph.write(uIdx + '\t' + iIdx + '\t' + tIdx + '\t' + str(w_UI) + '\t' + str(w_UT) + '\t' + str(w_IT) + '\n')