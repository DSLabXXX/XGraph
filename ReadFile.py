from scipy import *
from scipy.sparse import *


__author__ = 'IanKuo'


def read_uit_file(str_path, mat_x_ui, mat_x_ut, mat_x_it):

    """
    Read UIT graph.
    File schema: user \t item \t tag \t <UI> \t <UT> \t <IT>
    <UU> = f_UU_1,f_UU_2,...
    :param str_path: string as the file path to read graph data
    :param mat_x_ui: User-Item relation matrix (pre-allocated output)
    :param mat_x_ut: User-Tag relation matrix (pre-allocated output)
    :param mat_x_it: Item-Tag relation matrix (pre-allocated output)
    :return: null
    """

    val_feature_num_ui = 0
    val_feature_num_ut = 0
    val_feature_num_it = 0

    list_u = list()
    list_i = list()
    list_t = list()
    list_ui = list()
    list_ut = list()
    list_it = list()

    with open(str_path, 'r', encoding='UTF-8') as file_uit_graph:
        for line in file_uit_graph:
            l = line.strip('\n').split('\t')
            user = int(l[0])
            item = int(l[1])
            tag = int(l[2])
            f_ui = list(l[3].split(','))
            f_ut = list(l[4].split(','))
            f_it = list(l[5].split(','))

            list_u.append(user)
            list_i.append(item)
            list_t.append(tag)
    
            val_feature_num_ui = len(f_ui)
            val_feature_num_ut = len(f_ut)
            val_feature_num_it = len(f_it)
    
            while len(list_ui) < val_feature_num_ui:
                list_ui.append(list())
            while len(list_ut) < val_feature_num_ut:
                list_ut.append(list())
            while len(list_it) < val_feature_num_it:
                list_it.append(list())
    
            for i in range(val_feature_num_ui):
                list_ui[i].append(float(f_ui[i]))
    
            for i in range(val_feature_num_ut):
                list_ut[i].append(float(f_ut[i]))
    
            for i in range(val_feature_num_it):
                list_it[i].append(float(f_it[i]))

    val_user_num = max(list_u) + 1
    val_item_num = max(list_i) + 1
    val_tag_num = max(list_t) + 1

    for i in range(val_feature_num_ui):
        mat_x_ui.append(csr_matrix((array(list_ui[i]), (array(list_u), array(list_i))),
                                   shape=(val_user_num, val_item_num), dtype=float))

    for i in range(val_feature_num_ut):
        mat_x_ut.append(csr_matrix((array(list_ut[i]), (array(list_u), array(list_t))),
                                   shape=(val_user_num, val_tag_num), dtype=float))

    for i in range(val_feature_num_it):
        mat_x_it.append(csr_matrix((array(list_it[i]), (array(list_i), array(list_t))),
                                   shape=(val_item_num, val_tag_num), dtype=float))


def read_xx_file(str_path, mat_xx):

    """
    Read XX graph. (XX can be: UU, II and TT)
    :param str_path: string as the file path to read graph data
    :param mat_xx: X-X relation matrix (pre-allocated output)
    <UU> = f_UU_1,f_UU_2,...
    """

    val_feature_num_xx = len(mat_xx)

    with open(str_path, 'r', encoding='UTF-8') as file_xx_graph:
        for line in file_xx_graph:
            l = line.split('\t')
            user1 = int(l[0])
            user2 = int(l[1])
            f_uu = tuple(l[2].split(','))

            for i in range(val_feature_num_xx):
                mat_xx[i][user1, user2] += f_uu[i]
                mat_xx[i][user2, user1] += f_uu[i]