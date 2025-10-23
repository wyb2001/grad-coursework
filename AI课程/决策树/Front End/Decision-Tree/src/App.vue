<template>
  <div class="container">
    <h1>决策树可视化 - 预测心脏病</h1>

    <div class="flex-row">
      <div class="controls">
        <div class="file-upload">
          <label>上传CSV文件: </label>
          <input type="file" @change="handleFileUpload" accept=".csv">
        </div>

        <div class="parameters">
          <div>
            <label>最小分割样本数: </label>
            <input type="number" v-model.number="minSamplesSplit" min="2" max="20">
          </div>
          <div>
            <label>最大深度: </label>
            <input type="number" v-model.number="maxDepth" min="1" max="20">
          </div>
          <div>
            <label>测试集比例: </label>
            <input type="number" v-model.number="testRatio" min="0.1" max="0.5" step="0.1">
          </div>
        </div>

        <button @click="trainAndVisualize" :disabled="!fileData">训练并可视化</button>
        <div class="results" v-if="accuracy !== null">
          <p>准确率: {{ accuracy.toFixed(2) }}%</p>
          <p>训练样本数: {{ trainSize }}, 测试样本数: {{ testSize }}</p>
        </div>
      </div>

      <div class="controls">
        <!-- 年龄,性别,胸痛类型,静息血压,血清胆固醇,空腹血糖,静息心电图结果,最大心率,运动诱发心绞痛,ST段压低值,ST段峰值斜率,主要血管数量,地中海贫血 -->

        <div class="flex">
          <div class="parameters">
            <div>
              <label>年龄: </label>
              <input type="number" v-model.number="inputFeatures.age" min="1" max="120">
            </div>
            <div>
              <label>性别: </label>
              <select v-model.number="inputFeatures.sex">
                <option value="0">女性</option>
                <option value="1">男性</option>
              </select>
            </div>
            <div>
              <label>胸痛类型: </label>
              <select v-model.number="inputFeatures.cp">
                <option value="0">无症状</option>
                <option value="1">非心绞痛性疼痛</option>
                <option value="2">非典型心绞痛</option>
                <option value="3">典型心绞痛</option>
              </select>
            </div>
            <div>
              <label>静息血压 (mmHg): </label>
              <input type="number" v-model.number="inputFeatures.trestbps" min="60" max="200">
            </div>
            <div>
              <label>血清胆固醇 (mg/dl): </label>
              <input type="number" v-model.number="inputFeatures.chol" min="50" max="600">
            </div>
            <div>
              <label>空腹血糖: </label>
              <select v-model.number="inputFeatures.fbs">
                <option value="0">≤ 120 mg/dl</option>
                <option value="1">> 120 mg/dl</option>
              </select>
            </div>
          </div>

          <div class="parameters">
            <div>
              <label>静息心电图结果: </label>
              <select v-model.number="inputFeatures.restecg">
                <option value="0">正常</option>
                <option value="1">ST-T波异常</option>
                <option value="2">左心室肥厚</option>
              </select>
            </div>
            <div>
              <label>最大心率: </label>
              <input type="number" v-model.number="inputFeatures.thalach" min="30" max="220">
            </div>
            <div>
              <label>运动诱发心绞痛: </label>
              <select v-model.number="inputFeatures.exang">
                <option value="0">否</option>
                <option value="1">是</option>
              </select>
            </div>
            <div>
              <label>ST段压低值: </label>
              <input type="number" v-model.number="inputFeatures.oldpeak" min="0" max="10" step="0.1">
            </div>
            <div>
              <label>ST段峰值斜率: </label>
              <select v-model.number="inputFeatures.slope">
                <option value="0">下斜</option>
                <option value="1">平坦</option>
                <option value="2">上斜</option>
              </select>
            </div>
            <div>
              <label>荧光透视着色的血管数量: </label>
              <select v-model.number="inputFeatures.ca">
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
              </select>
            </div>
            <div>
              <label>地中海贫血: </label>
              <select v-model.number="inputFeatures.thal">
                <option value="1">正常</option>
                <option value="2">固定缺陷</option>
                <option value="3">可逆缺陷</option>
              </select>
            </div>
          </div>

        </div>
        <button @click="predictionDisease">查询</button>
      </div>

      <div v-if="isSearch">
        <div class="predict-safe" v-if="sState == 0">
          <p>此人的病理路径: {{ }}</p>
          <p v-for="node in this.tree.search_path">
            {{ node.feature !== null ? `${node.titles} ${node.direction === 'left' ? '<=' : '>'} ${node.threshold_name}` :
              `预测类别: ${this.tree.diag['心脏病诊断结果'][node.value]}` }} </p>
        </div>
        <div class="predict-danger" v-if="sState == 1">
          <p>此人的病理路径: {{ }}</p>
          <p v-for="node in this.tree.search_path">
            {{ node.feature !== null ? `${node.titles} ${node.direction === 'left' ? '<=' : '>'} ${node.threshold_name}` :
              `预测类别: ${this.tree.diag['心脏病诊断结果'][node.value]}` }} </p>
        </div>

      </div>


    </div>

    <div class="tree-container" v-if="richMediaData">
      <vue-tree style="width: 100%; height: 800px; border: 1px solid gray;" :dataset="richMediaData"
        :config="treeConfig">
        <template v-slot:node="{ node, collapsed }">
          <div class="rich-media-node" :style="{
            border: collapsed ? '2px solid grey' : '',
            backgroundColor: node.isLeaf ? '#e8f5e8' : '#f0f8ff'
          }">
            <div v-if="node.isLeaf" style="text-align: center;">
              <div style="font-weight: bold; color: #2e7d32;">叶节点</div>
              <div>预测类别: {{ node.name }}</div>
            </div>
            <div v-else style="text-align: center;">
              <div style="font-weight: bold; color: #1565c0;">决策节点</div>
              <div> {{ node.feature }} <= {{ node.threshold_name }}</div>
            </div>
          </div>
        </template>
      </vue-tree>
    </div>

    <div v-if="errorMessage" class="error">
      {{ errorMessage }}
    </div>
  </div>
</template>

<script>
import { isThisTypeNode } from 'typescript';

class Node {
  constructor(feature = null, threshold = null, left = null, right = null, value = null) {
    this.feature = feature;
    this.threshold = threshold;
    this.threshold_name = null;
    this.left = left;
    this.right = right;
    this.value = value;
  }

  isLeaf() {
    return this.value !== null;
  }
}

class DecisionTree {
  constructor(minSamplesSplit = 2, maxDepth = 10, titles = null) {
    this.minSamplesSplit = minSamplesSplit;
    this.maxDepth = maxDepth;
    this.root = null;
    this.titles = titles;
    this.search_path = [];
    this.diag = {
      '性别': ['女性', '男性'],
      '胸痛类型': ['无症状', '轻微', '非心源性疼痛', '典型心绞痛'],
      '静息心电图结果': ['ST-T波异常', '正常', '显示可能或确定的左心室肥大'],
      '运动诱发心绞痛': ['否', '是'],
      'ST段峰值斜率': ['上升', '平坦', '下降'],
      '地中海贫血': ['_', '正常', '轻度', '重度'],
      '心脏病诊断结果': ['无心脏病', '有心脏病']
    }
  }

  fit(x, y) {
    this.root = this._create(x, y, 0);
  }

  _gini(y) {
    const counter = {};
    for (const val of y) {
      counter[val] = (counter[val] || 0) + 1;
    }
    const total = y.length;
    let gini = 1;
    for (const count of Object.values(counter)) {
      gini -= (count / total) ** 2;
    }
    return gini;
  }

  _findThreshold(x, y) {
    let bestGini = Infinity;
    let bestFeature = null;
    let bestThreshold = null;
    const m = x.length;
    const n = x[0].length;

    for (let feature = 0; feature < n; feature++) {
      const featureValues = [...new Set(x.map(row => row[feature]))].sort((a, b) => a - b);

      for (const threshold of featureValues) {
        const leftMask = x.map(row => row[feature] <= threshold);
        const rightMask = leftMask.map(val => !val);

        const leftY = y.filter((_, i) => leftMask[i]);
        const rightY = y.filter((_, i) => rightMask[i]);

        if (leftY.length === 0 || rightY.length === 0) continue;

        const giniLeft = this._gini(leftY);
        const giniRight = this._gini(rightY);
        const weightedGini = (leftY.length * giniLeft + rightY.length * giniRight) / m;

        if (weightedGini < bestGini) {
          bestGini = weightedGini;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    return [bestFeature, bestThreshold];
  }

  _create(x, y, depth) {
    const dataNum = x.length;
    const uniqueClasses = [...new Set(y)];

    if (dataNum < this.minSamplesSplit || depth >= this.maxDepth || uniqueClasses.length === 1) {
      const counter = {};
      for (const val of y) {
        counter[val] = (counter[val] || 0) + 1;
      }
      const mostCommon = Object.entries(counter).reduce((a, b) => a[1] > b[1] ? a : b)[0];
      return new Node(null, null, null, null, parseInt(mostCommon));
    }

    const [feature, threshold] = this._findThreshold(x, y);

    if (feature === null) {
      const counter = {};
      for (const val of y) {
        counter[val] = (counter[val] || 0) + 1;
      }
      const mostCommon = Object.entries(counter).reduce((a, b) => a[1] > b[1] ? a : b)[0];
      return new Node(null, null, null, null, parseInt(mostCommon));
    }

    const leftMask = x.map(row => row[feature] <= threshold);
    const rightMask = leftMask.map(val => !val);

    const leftX = x.filter((_, i) => leftMask[i]);
    const leftY = y.filter((_, i) => leftMask[i]);
    const rightX = x.filter((_, i) => rightMask[i]);
    const rightY = y.filter((_, i) => rightMask[i]);

    const left = this._create(leftX, leftY, depth + 1);
    const right = this._create(rightX, rightY, depth + 1);

    return new Node(feature, threshold, left, right, null);
  }

  predict(x) {
    return x.map(sample => this._search(sample, this.root));
  }

  search(x) {
    this.search_path = [];
    return this._search_person(x);
  }

  _search_person(x, tree = this.root) {
    if (tree.isLeaf()) {
      this.search_path.push({ feature: null, threshold: null, direction: 'leaf', value: tree.value });
      return tree.value;
    }
    if (x[tree.feature] > tree.threshold) {
      this.search_path.push({ feature: tree.feature, titles: this.titles[tree.feature], threshold: tree.threshold, direction: 'right', threshold_name: tree.threshold_name });
      return this._search_person(x, tree.right);
    } else {
      this.search_path.push({ feature: tree.feature, titles: this.titles[tree.feature], threshold: tree.threshold, direction: 'left', threshold_name: tree.threshold_name});
      return this._search_person(x, tree.left);
    }
  }

  _search(x, tree) {
    if (tree.isLeaf()) {
      return tree.value;
    }
    if (x[tree.feature] > tree.threshold) {
      return this._search(x, tree.right);
    } else {
      return this._search(x, tree.left);
    }
  }

  // 将决策树转换为可视化数据格式
  toVueTreeData() {
    return this._convertNode(this.root);
  }

  _convertNode(node, sampleCount = 100) {
    if (node.isLeaf()) {
      return {
        name: this.diag['心脏病诊断结果'][node.value],
        value: sampleCount,
        isLeaf: true,
        feature: null,
        threshold: null
      };
    }
    else {
      if (this.titles[node.feature] == '性别' ||
        this.titles[node.feature] == '胸痛类型' ||
        this.titles[node.feature] == '静息心电图结果' ||
        this.titles[node.feature] == '运动诱发心绞痛' ||
        this.titles[node.feature] == 'ST段峰值斜率' ||
        this.titles[node.feature] == '地中海贫血'
      ) {
        node.threshold_name = this.diag[this.titles[node.feature]][node.threshold];
      }
      else {
        node.threshold_name = node.threshold;
      }

      return {
        name: ` ${this.titles[node.feature]}`,
        value: sampleCount,
        isLeaf: false,
        feature: this.titles[node.feature],
        threshold: node.threshold,
        threshold_name: node.threshold_name,
        children: [
          this._convertNode(node.left, Math.floor(sampleCount * 0.6)),
          this._convertNode(node.right, Math.floor(sampleCount * 0.4))
        ]
      };
    }
  }
}

export default {
  name: 'TreeVisualization',
  data() {
    return {
      minSamplesSplit: 2,
      maxDepth: 10,
      testRatio: 0.2,
      accuracy: null,
      isSearch: null,
      sState: 0,
      trainSize: 0,
      testSize: 0,
      richMediaData: null,
      fileData: null,
      tree: null,
      fileName: '',
      errorMessage: '',
      treeConfig: {
        nodeWidth: 160,
        nodeHeight: 80,
        levelHeight: 100
      },
      //<!-- 年龄,性别,胸痛类型,静息血压,血清胆固醇,空腹血糖,静息心电图结果,最大心率,运动诱发心绞痛,ST段压低值,ST段峰值斜率,主要血管数量,地中海贫血 -->
      featureNames: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
      // 输入特征默认值
      inputFeatures: {
        age: 50,
        sex: 0,
        cp: 0,
        trestbps: 120,
        chol: 200,
        fbs: 0,
        restecg: 0,
        thalach: 150,
        exang: 0,
        oldpeak: 1.0,
        slope: 1,
        ca: 0,
        thal: 2
      }
    }
  },
  methods: {
    handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      this.fileName = file.name;
      this.errorMessage = '';

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csvContent = e.target.result;
          this.parseCSV(csvContent);
        } catch (error) {
          this.errorMessage = `文件解析错误: ${error.message}`;
        }
      };
      reader.readAsText(file);
    },

    parseCSV(csvContent) {
      const lines = csvContent.split('\n').filter(line => line.trim() !== '');

      if (lines.length < 2) {
        this.errorMessage = 'CSV文件内容过少，至少需要一行表头和一行数据';
        return;
      }

      // 解析表头
      const headers = this.parseCSVLine(lines[0]);

      // 解析数据
      const data = [];
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        const row = this.parseCSVLine(line);

        // 转换为数值
        const numericRow = row.map(cell => {
          const num = parseFloat(cell);
          return isNaN(num) ? cell : num;
        });

        data.push(numericRow);
      }

      this.fileData = data;
      this.fileHeaders = headers; // 保存表头，可能用于显示
    },

    parseCSVLine(line) {
      // 简单的CSV解析，处理引号包围的字段
      const result = [];
      let current = '';
      let inQuotes = false;

      for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          result.push(current);
          current = '';
        } else {
          current += char;
        }
      }

      result.push(current);
      return result;
    },

    trainAndVisualize() {
      if (!this.fileData) {
        this.errorMessage = '请先上传CSV文件';
        return;
      }

      try {
        const data = this.fileData;
        const tot = data.length;
        const testLen = Math.floor(this.testRatio * tot);

        // 随机打乱数据
        const shuffled = [...data].sort(() => Math.random() - 0.5);

        const testX = shuffled.slice(0, testLen).map(row => row.slice(0, -1));
        const testY = shuffled.slice(0, testLen).map(row => row[row.length - 1]);
        const trainX = shuffled.slice(testLen).map(row => row.slice(0, -1));
        const trainY = shuffled.slice(testLen).map(row => row[row.length - 1]);

        this.trainSize = trainX.length;
        this.testSize = testX.length;

        // 训练决策树
        this.tree = new DecisionTree(this.minSamplesSplit, this.maxDepth, this.fileHeaders);
        this.tree.fit(trainX, trainY);

        // 预测并计算准确率
        const predictions = this.tree.predict(testX);
        const correct = predictions.reduce((count, pred, i) =>
          count + (pred === testY[i] ? 1 : 0), 0);
        this.accuracy = (correct / testLen) * 100;

        // 转换为可视化数据
        this.richMediaData = this.tree.toVueTreeData();
        this.errorMessage = '';
      } catch (error) {
        this.errorMessage = `训练过程中出错: ${error.message}`;
      }
    },

    predictionDisease() {
      // 将inputFeatures按照featureNames的顺序转换为数组
      const features = this.featureNames.map(name => this.inputFeatures[name]);
      // 检查是否有未输入的特征
      if (features.some(f => f === null || f === '')) {
        this.errorMessage = '请填写所有特征值';
        return;
      }

      // 预测
      this.sState = this.tree.search(features); // 0 / 1
      this.isSearch = true;
    }
  }
}
</script>

<style scoped>
.container {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.flex-row {
  /* 方法1: 使用flexbox (推荐，现代布局方式) */
  display: flex;
  margin: 20px;
  gap: 100px;
  border: 1px dashed #ccc;
  /* 控制子元素之间的间距 */
}

.flex {
  display: flex;
  gap: 30px;
}

.controls {
  margin: 20px 10px;
  padding: 30px;
  border: 1px solid #ddd;
  border-radius: 5px;
  background-color: #f9f9f9;
}

.predict-safe {
  margin: 20px 10px;
  padding: 30px;
  background-color: #e8f5e8;
  border: 1px solid #25ac29;
  border-radius: 4px;
}

.predict-danger {
  margin: 20px 10px;
  padding: 30px;
  background-color: #e8453d;
  border: 1px solid #ec5c5c;
  border-radius: 4px;
}

.file-upload {
  margin-bottom: 15px;
}

.file-upload label {
  font-weight: bold;
}

.file-name {
  margin-left: 10px;
  color: #4CAF50;
  font-weight: bold;
}

.parameters div {
  margin: 10px 0;
}

.parameters label {
  display: inline-block;
  width: 150px;
  font-weight: bold;
}

.parameters input {
  padding: 5px;
  border: 1px solid #ccc;
  border-radius: 3px;
  width: 80px;
}

button {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin-top: 10px;
}

button:hover:not(:disabled) {
  background-color: #45a049;
}

button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.results {
  margin: 15px 0;
  padding: 10px;
  background-color: #e8f5e8;
  border: 1px solid #4CAF50;
  border-radius: 4px;
}

.results p {
  margin: 5px 0;
  font-weight: bold;
  color: #2e7d32;
}

.tree-container {
  margin-top: 20px;
}

.rich-media-node {
  padding: 8px;
  border-radius: 6px;
  border: 2px solid #2196F3;
  text-align: center;
  font-size: 12px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.error {
  margin: 15px 0;
  padding: 10px;
  background-color: #ffebee;
  border: 1px solid #f44336;
  border-radius: 4px;
  color: #c62828;
  font-weight: bold;
}
</style>