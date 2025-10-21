import { createApp } from 'vue'
import App from './App.vue'

// 导入树形组件
import VueTree from "@ssthouse/vue3-tree-chart";
import "@ssthouse/vue3-tree-chart/dist/vue3-tree-chart.css";

const app = createApp(App)

// 全局注册组件
app.component('VueTree', VueTree)

app.mount('#app')