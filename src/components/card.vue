<template>
<div class="card">
    <div class="content">  

        <h3>{{title}}</h3>
        <p>{{explain}}</p>
        <template v-for="(option, index) in options" :key="index">
            <label class="list-item" v-if="isMultiple">
                <input @click="returnSelection(option)" type="checkbox" :name="id" /> {{ option }}
            </label>
            <label class="list-item" v-else>
                <input @click="returnSelection(option)" type="radio" :name="id" /> {{ option }}
            </label>
        </template>

    </div>
</div>
</template>

<script>
export default{
    data(){
        return{
            selection: [], // 保存用户选择的选项值
        }
    },
    props:{
        id:{
            type:Number
        },
        title:{
            type:String
        },
        explain:{
            type:String
        },
        isMultiple:{
            type:Boolean
        },
        options:{
            type:Array
        }
    },

    methods:{
        returnSelection(option) {
            if (this.isMultiple) {
            // 如果是多选，将选项值添加到已选择的选项字符串中
                if (this.selection.includes(option)) {
                    // 已选择的选项中已经包含该选项，移除它
                    const index = this.selection.indexOf(option);
                    this.selection.splice(index, 1);
                } else {
                    // 选择一个新选项，添加它
                    this.selection.push(option);
                }
            } else {
                // 单选，直接保存选项值
                this.selection = [option];
            }
            //console.log(this.selection)
            // 触发自定义事件，将选项值传递给父组件
            this.$emit('option-selected', this.selection,this.id,this); 
            this.$emit('type-selected', this.selection); 
        }
    }
}
</script>

<style scoped>
.card{
    /* 相对定位 */
    position: relative;
    width: 300px;
    height: 400px;
    background-color: rgba(255,255,255,0.1);
    margin: 30px;
    border-radius: 15px;
    /* 阴影 */
    box-shadow: 20px 20px 50px rgba(0,0,0,0.5);
    /* 溢出滚动 */
    overflow: auto;
    display: flex;
    justify-content: center;
    align-self: start;
    border-top: 1px solid rgba(255,255,255,0.5);
    border-left: 1px solid rgba(255,255,255,0.5);
    /* 背景模糊 */
    backdrop-filter: blur(5px);
}
/* 滚动条设置 */
.card::-webkit-scrollbar {
    width: 5px;
    height: 200px;
    background-color: #201d2a;
}
.card::-webkit-scrollbar-track {
    -webkit-box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
    border-radius: 1px;
    background-color: #201d2a;
}
.card::-webkit-scrollbar-thumb {
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.3);
}
.card .content{
    padding: 20px;
    text-align: center;
}
.card .content h3{
    font-size: 28px;
    color: #fff;
}
.card .content p{
    font-size: 13px;
    color: #fff;
    font-weight: 300;
    margin: 10px 0 15px 0;
}
.list-item {
    display: flex;
    vertical-align:middle;
    font-size: 18px;
    color: #fff;
    font-weight: 300;
    margin: 20px 0 15px 0;
}
.list-item input[type="radio"] {
    margin-right: 10px;
    zoom: 130%;
}
.list-item input[type="checkbox"] {
    margin-right: 10px;
    zoom: 130%;
}
</style>