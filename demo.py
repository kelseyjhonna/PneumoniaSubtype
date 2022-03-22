import pandas as pd
import matplotlib
import seaborn as sns
import warnings
from PIL import Image
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap
import os
import streamlit.components.v1 as components

# 生存分析
from lifelines import AalenAdditiveFitter
from lifelines import KaplanMeierFitter


sns.set(style='ticks')
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='FangSong', weight='bold')  # 用于画图时显示中文

warnings.filterwarnings('ignore')


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


@st.cache(suppress_st_warning=True)
def dis_img1():
    image1 = Image.open('./pneu_img/聚类散点密度图2.png')  # Image name
    # fig = plt.figure()
    # plt.imshow(image1)
    # plt.axis("off")

    image2 = Image.open('./pneu_img/Subphenotype_Status1.tif')  # Image name
    # fig = plt.figure()
    # plt.imshow(image2)
    # plt.axis("off")
    return image1, image2


image1, image2 = dis_img1()

#
# @st.cache(suppress_st_warning=True)
# def dis_img2():
#     image1 = Image.open('./pneu_img/流程图.png')  # Image name
#     fig = plt.figure()
#     plt.imshow(image1)
#     plt.axis("off")
#     return image1
#
#
# image3 = dis_img2()


def show_dashbord():
    # st.markdown(
    #     ":point_left: **打开右侧栏并输入健康指标使用肺炎亚型识别工具** :grey_exclamation:")

    st.title('重症肺炎亚型识别与生存分析')
    st.markdown(
        '我们基于 [MIMIC-IV](https://mimic.mit.edu/) 开发了重症肺炎亚型识别模型，'
        '区分重症肺炎患者亚型，并进一步指导临床医生提供适当的治疗，以期提高患者生存率。')
    col1, col2 = st.columns([2, 3])
    with col1:
        resize_image1 = image1.resize((300, 280))
        st.image(resize_image1, caption="Clustering Distribution")
        st.markdown('研究结果表明，重症肺炎患者可以分为两种临床亚型。')
        # st.markdown('\n \n')
    with col2:
        resize_image2 = image2.resize((680, 423))
        st.image(resize_image2, caption="Number of Subtypes and Survival Probability")
        st.markdown(
            '两种亚型之间死亡率具有显着差异。\n亚型1比亚型2的患者幸存率高7.6%!')

        # st.image(image3, caption="Flow Chart")
        # st.markdown('Retrospective analysis showed that there were also significant differences '
        #             'in medication between the two subtypes, which further proved the availability of '
        #             'this subtype recognition model. This is described in more detail in our paper.')
    st.markdown('--')
    st.markdown(
        ":point_left: **打开右侧栏并输入健康指标使用肺炎亚型识别工具** :grey_exclamation:")


st.sidebar.title("请输入您的健康指标: ")

age = st.sidebar.number_input('年龄 (岁): ', min_value=18, max_value=89,value=18, step=1)
temp = st.sidebar.number_input('体温 (°F): ', min_value=96.1, max_value=100.4,value=97.3, step=0.1)
rr = st.sidebar.number_input('呼吸频率 (insp/min): ', min_value=4, max_value=36,value=23, step=1)
po2 = st.sidebar.number_input('血氧分压 (mmHg): ', min_value=8, max_value=361,value=322, step=1)
pco2 = st.sidebar.number_input('动脉二氧化碳分压 (mmHg): ', min_value=13.5, max_value=33.0, value=23.5, step=0.5)
o2 = st.sidebar.number_input('血氧饱和度 (%): ', min_value=88, max_value=100,value=88, step=1)
tco2 = st.sidebar.number_input('动脉二氧化碳总量 (mEq/L): ', min_value=12, max_value=39, value=22, step=1)
ao2 = st.sidebar.number_input('动脉氧分压 (mmHg): ', min_value=88, max_value=100,value=88, step=1)
o2f = st.sidebar.number_input('氧气流速 (L/min): ', min_value=0.5, max_value=20.0, value=6.0, step=0.5)
chloride = st.sidebar.number_input('氯离子 (mEq/L): ', min_value=86, max_value=121, value=98, step=1)
bicarbonate = st.sidebar.number_input('碳酸氢盐 (mEq/L): ', min_value=13, max_value=33, value=26, step=1)
sodium = st.sidebar.number_input('钠离子 (mEq/L): ', min_value=129, max_value=148, value=134, step=1)
aniongap = st.sidebar.number_input('阴离子间隙: ', min_value=5, max_value=24, value=12, step=1)
glucose = st.sidebar.number_input('血糖 (mg/dL): ', min_value=14, max_value=254,value=189, step=1)
albumin = st.sidebar.number_input('白蛋白 (g/dL): ', min_value=1.2, max_value=4.7, value=2.6, step=0.1)
lymphoctyes = st.sidebar.number_input('淋巴细胞 (%): ', min_value=0.4, max_value=30.3, value=15.5, step=0.1)
wbc = st.sidebar.number_input('白细胞 (K/uL): ', min_value=0.1, max_value=26.7, value=19.8, step=0.1)
rdw = st.sidebar.number_input('红细胞分布宽度 (%): ',  min_value=11.1, max_value=21.5, value=13.3,step=0.1)
platelet = st.sidebar.number_input('血小板计数 (K/uL) ',  min_value=5, max_value=469,value=144,step=1)
ld = st.sidebar.number_input('乳酸脱氢酶 (IU/L): ',  min_value=60, max_value=709, value=216,step=1)
plr = st.sidebar.number_input('疼痛级别响应 (%): ', min_value=1, max_value=7, value=5, step=1)

FirstCareUnit = st.sidebar.selectbox(
     '第一个重症监护室？',
     ('冠心病/心血管重症监护病房', '内科/外科重症监护病房', '神经外科重症监护病房', '胸外科重症监护病房'))

st.sidebar.markdown("\n")
st.sidebar.markdown("是否有如下情况: ")
invent = st.sidebar.checkbox('使用有创呼吸机？', value=False)
can = st.sidebar.checkbox('有癌症？', value=False)
Cere = st.sidebar.checkbox('有脑血管疾病？', value=False)
bs = st.sidebar.checkbox('血痰培养异常？', value=False)

apsiii = st.sidebar.slider('APSIII (急性生理学评分): ', min_value=0, max_value=184, value=145, step=1)
sofa = st.sidebar.slider('SOFA (序贯器官衰竭评分): ', min_value=0, max_value=19, value=4, step=1)

predict = st.sidebar.button("预测")
back = st.sidebar.button("返回")


if predict:
    # ['BMI', '220179', '220235', 'hypertension', '223761', 'cancer', '51301',
    #   '50902', '50931', 'age', '220210', '50882', '224409', '50983', '50821',
    #   'BLOODSPUTUM', '50818', 'apsiii', '50862', 'cerebrovascular_disease',
    #   '223834', 'sofa', '220277', 'InvasiveVent']
    fcu_list = ['冠心病/心血管重症监护病房', '内科/外科重症监护病房', '神经外科重症监护病房', '胸外科重症监护病房']
    fcu = fcu_list.index(FirstCareUnit)

    InvasiveVentilator, Microbiological, Cancer, Cerebrovascular = 0, 0, 0, 0
    if invent: InvasiveVentilator = 1
    if Cere: Cerebrovascular = 1
    if can: Cancer = 1
    if bs: Microbiological = 1

    data_list = [age, temp, rr, po2, pco2, o2, tco2, ao2, o2f, chloride,
                 bicarbonate, sodium, aniongap, glucose, albumin, lymphoctyes, wbc, rdw, platelet, ld,
                 plr, fcu, InvasiveVentilator, Microbiological, Cancer, Cerebrovascular, apsiii, sofa]
    # 28个变量
    data_df = pd.DataFrame(np.array(data_list)).T

    info_df = pd.read_csv('./Datasets/mean_std_df.csv', encoding='gbk')

    # 如果需要标准化
    # data_df = pd.DataFrame((np.array(data_list) - info_df['mean']) / info_df['std']).T

    # 修改列名
    data_df.columns = info_df.Variable_ch.to_list()
    # print(data_df)
    # print(data_df.shape)

    st.markdown("## :bell: 亚型预测结果:")

    # st.dataframe(data_df)
    # st.write(data_df.shape)

    # ['Temperature Fahrenheit', 'Cancer', 'White Blood Cells', 'Chloride',
    #  'Glucose', 'Age', 'Respiratory Rate', 'Bicarbonate',
    #  'Pain Level Response', 'Sodium', 'pO2', 'Blood Sputum', 'pCO2',
    #  'APSIII', 'Albumin', 'Cerebrovascular Disease', 'O2 Flow', 'SOFA',
    #  'O2 saturation pulseoxymetry', 'Invasive Ventilator']

    sub_list = ['体温', '癌症', '白细胞', '氯离子', '血糖',
                '年龄', '呼吸频率', '碳酸氢盐', '疼痛级别响应', '钠离子',
                '动脉氧分压', '血痰培养', '动脉二氧化碳分压', 'APSIII', '白蛋白',
                '脑血管疾病', '氧气流速', 'SOFA', '血氧饱和度', '有创呼吸机']
    df_sub = data_df.filter(sub_list)
    # df_sub.to_scv('./test.csv', encoding='gbk')
    gbc_model = load_model('./Models/GBC.pkl')
    proba = gbc_model.predict_proba(df_sub)

    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        st.write(":point_right: 属于亚型 :one: 的概率是: ")
        st.markdown("## **%.3f**" % proba[0, 0])
    with col3:
        st.write(":point_right: 属于亚型 :two: 的概率是: ")
        st.markdown("## **%.3f**" % proba[0, 1])

    st.markdown("### 预测结果解释:")
    explainer = load_model('./Models/shap_explainer.pkl')
    shap_values = explainer.shap_values(df_sub)

    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0],
                    df_sub.iloc[0].values,
                    feature_names=df_sub.columns.to_list(), link='logit')
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(shap_html)
    st.markdown('依赖图详细的展示了促使患者分类为某个亚型的具体的指标作用。')
    st.markdown('红色表示指标驱使患者分类为亚型 :one: ，蓝色表示指标驱使患者分类为亚型 :two: 。')
    st.markdown('越靠近颜色分界中心的变量作用性越强。')

    # ------------------------------------- 生存分析 --------------------------
    st.markdown("## :bell: 生存分析结果:")
    df_surv = data_df.copy()
    df_surv['sub_proba'] = proba[0, 1]

    df_surv.columns = ['Age', 'TemperatureFahrenheit', 'RespiratoryRate', 'pO2', 'pCO2',
                       'O2saturationpulseoxymetry', 'TCO2calcArterial', 'ArterialO2pressure',
                       'O2Flow', 'Chloride', 'Bicarbonate', 'Sodium', 'Aniongap', 'Glucose',
                       'Albumin', 'Lymphocytes', 'WhiteBloodCells', 'RDW', 'PlateletCount',
                       'LactateDehydrogenaseLD', 'PainLevelResponse', 'FirstCareUnit',
                       'InvasiveVentilator', 'BloodSputum', 'Cancer', 'CerebrovascularDisease',
                       'APSIII', 'SOFA', 'Score1']
    # st.write(df_surv)
    # st.write(df_surv.shape)

    KM_df = pd.read_csv('./Datasets/KM_df.csv')
    aaf = load_model('./Models/AAF.pkl')

    # 分组 : K-M 存活曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    kmf = KaplanMeierFitter()

    name_lsit = ['亚型1', '亚型2']
    color_list = ['#fb9d50', '#6aadd5']

    for name, grouped_df in KM_df.groupby('Subphenotype'):
        kmf.fit(grouped_df["HospTime"], grouped_df["HospStatus"], label=name_lsit[name])
        kmf.plot_survival_function(ax=ax, color=color_list[name])

    X = df_surv.loc[0]
    # st.write(X)
    aaf.predict_survival_function(X).rename(columns={0: '患者预测'}).plot(ax=ax)

    ax.set_xlim([0, 40])
    ax.set_ylim([0.5, 1])
    ax.set_xlabel('时间（天）', fontsize=20)
    ax.set_ylabel('生存概率', fontsize=20)
    ax.tick_params(which='major', labelsize=18)
    ax.legend(fontsize=18)
    ax.grid(True)
    st.pyplot(fig, dpi=600)

    st.write('生存概率曲线展示了患者当前状态下随着时间推移的生存预测情况，黄色线表示亚型 :one: 的总体预测，蓝色线表示亚型 :two: 的总体预测，绿色线表示当前患者的预测生存概率曲线。')

else:
    show_dashbord()

if back and predict:
    show_dashbord()
