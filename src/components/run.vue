<template>
<!-- 按钮设置 -->
<div>
    <button class="button">
        <strong @click="output()">RUN</strong>
        <span></span>
    </button>
</div>
<!--运行结果输出栏-->
<div class="run_input">
    <div class="content">
        <p v-if="ifTrain">&nbsp;&nbsp;Training&nbsp;&nbsp;.&nbsp;&nbsp;.&nbsp;&nbsp;.</p>
        <ul>
            <li v-for="(item, index) in result" :key="index">&nbsp;&nbsp;{{ item }}</li>
        </ul>
        <div id="imageContainer"></div>
        <!-- <p>预测值与真实值对比散点图</p>   -->
    </div>
</div>    
</template>

<script>
export default{
    emits: ['click-send'], // 声明可以触发的事件

    data(){
        return{
            result: [],
            ifTrain:false
        }
    },
    props:{
        res:{
            type:Array
        }
    },
    watch: {
        res(newRes) {
            //console.log(newRes)
            if(Object.keys(newRes).length !== 0){
                // 更新子组件的显示内容
                this.ifTrain=false;
                //console.log("ewRes")
            }
                // newResult 是 result 的新值
                this.result=newRes;
        }
    },
    methods:{
        output(){
            this.ifTrain=true;
            this.$emit('click-send'); 
        }
    }
}
</script>

<style scoped>
/* 按钮样式 */
.button {    
    position: relative;
    border: none;
    margin:50px 600px 30px 600px;
    display: inline-block;
    position: relative;
    padding: 0.5em 2.2em;
    font-size: 25px;
    font-weight:bolder;
    background: transparent;
    /* 背景透明 */
    cursor: pointer;
    user-select: none;
    /* color: royalblue; */
    color: rgb(65, 159, 241);
    overflow: hidden;
    /* 溢出隐藏 */
    outline: none;
    /* 隐藏边框 */
    cursor: pointer;
}

.button span {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: transparent;
    z-index: -1;
    border: 5px solid rgb(65, 159, 241);
}

.button span::before {
    content: '';
    position: absolute;
    width: 8%;
    height: 500%;
    background: #161626;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-60deg);
    transition: all .3s;
}

.button:hover span::before {
    transform: translate(-50%, -50%) rotate(-90deg);
    width: 100%;
    background: rgb(65, 159, 241);
}

.button:hover {
    color: white;
}

/* 点击聚焦 */
.button:active:focus {
    opacity: 0.9;
    /* 透明 */
}

/* 运行输出框 */
.run_input{
    /* 相对定位 */
    position: relative;
    width: 1000px;
    height: auto;
    background-color: rgba(255,255,255,0.1);
    margin: 30px;
    border-radius: 15px;
    /* 阴影 */
    box-shadow: 20px 20px 50px rgba(0,0,0,0.5);
    /* 溢出滚动 */
    
    display: flex;
    justify-content: center;
    align-self: start;
    border-top: 1px solid rgba(255,255,255,0.5);
    border-left: 1px solid rgba(255,255,255,0.5);
    /* 背景模糊 */
    backdrop-filter: blur(5px);
}
.run_input li{
    font-size: 20px;
    color: #fff;
    font-weight: 300;
    margin: 35px 0 35px 0;
}
.run_input p{
    font-size: 20px;
    color: #fff;
    font-weight: 300;
    margin: 35px 0 35px 0;
}
#imageContainer{
    max-width: 10%;

}
</style>