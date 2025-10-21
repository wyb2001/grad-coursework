<template>
  <div class="container">
    <h1>决策树可视化</h1>
    
    <div class="controls">
      <div class="file-upload">
        <label>上传CSV文件: </label>
        <input type="file" @change="handleFileUpload" accept=".csv">
        <span v-if="fileName" class="file-name">{{ fileName }}</span>
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
    </div>
    
    <div class="results" v-if="accuracy !== null">
      <p>准确率: {{ accuracy.toFixed(2) }}%</p>
      <p>训练样本数: {{ trainSize }}, 测试样本数: {{ testSize }}</p>
    </div>

    <div class="tree-container" v-if="richMediaData">
      <vue-tree
        style="width: 100%; height: 800px; border: 1px solid gray;"
        :dataset="richMediaData"
        :config="treeConfig"
      >
        <template v-slot:node="{ node, collapsed }">
          <div
            class="rich-media-node"
            :style="{ 
              border: collapsed ? '2px solid grey' : '',
              backgroundColor: node.isLeaf ? '#e8f5e8' : '#f0f8ff'
            }"
          >
            <div v-if="node.isLeaf" style="text-align: center;">
              <div style="font-weight: bold; color: #2e7d32;">叶节点</div>
              <div>预测类别: {{ node.name }}</div>
            </div>
            <div v-else style="text-align: center;">
              <div style="font-weight: bold; color: #1565c0;">决策节点</div>
              <div> {{ node.feature }} ≤ {{ node.threshold }}</div>
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
class Node {
  constructor(feature = null, threshold = null, left = null, right = null, value = null) {
    this.feature = feature;
    this.threshold = threshold;
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
    this.diag = {
      '性别': ['女性', '男性'],
      '胸痛类型': ['无症状', '轻微', '非心源性疼痛', '典型心绞痛'],
      '静息心电图结果': ['ST-T波异常', '正常', '显示可能或确定的左心室肥大'],
      '运动诱发心绞痛': ['否', '是'],
      'ST段峰值斜率': ['上升', '平坦', '下降'],
      '地中海贫血': ['_','正常', '轻度', '重度'],
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
        node.threshold = this.diag[this.titles[node.feature]][node.threshold];
      }

        return {
          name: ` ${this.titles[node.feature]}`,
          value: sampleCount,
          isLeaf: false,
          feature: this.titles[node.feature],
          threshold: node.threshold,
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
      trainSize: 0,
      testSize: 0,
      richMediaData: null,
      fileData: null,
      fileName: '',
      errorMessage: '',
      treeConfig: { 
        nodeWidth: 160, 
        nodeHeight: 80, 
        levelHeight: 100 
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
        const tree = new DecisionTree(this.minSamplesSplit, this.maxDepth, this.fileHeaders);
        tree.fit(trainX, trainY);

        // 预测并计算准确率
        const predictions = tree.predict(testX);
        const correct = predictions.reduce((count, pred, i) => 
          count + (pred === testY[i] ? 1 : 0), 0);
        this.accuracy = (correct / testLen) * 100;

        // 转换为可视化数据
        this.richMediaData = tree.toVueTreeData();
        this.errorMessage = '';
      } catch (error) {
        this.errorMessage = `训练过程中出错: ${error.message}`;
      }
    }
  }
}
</script>

<style scoped>
.container {
  padding: 20px;
  font-family: Arial, sans-serif;
}

.controls {
  margin: 20px 0;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 5px;
  background-color: #f9f9f9;
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
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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