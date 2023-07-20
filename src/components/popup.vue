<template>
    <div class="popup">

      <!-- 弹窗内容 -->
      <label class="content">
        <template v-for="(option, index) in parameter" :key="index">
          {{ option }}<input type="text" v-model="inputValues[index]" />
        </template>
      </label>

      <button class="close-button" @click="closePopup">确定</button>
    </div>
</template>
  
<script>
export default {
  data() {
    return {
      inputValues: {}
    }
  },
  props: {
    parameter:{
      type:Array
    },
  },
  methods: {
    closePopup() {
      const values = {};
      for (let i = 0; i < this.parameter.length; i++) {
        const option = this.parameter[i];
        values[option] = this.inputValues[i];
      }
      //console.log(values)
      this.$emit('saveValues',values);
    }
  },
}
</script>
  
<style scoped>
  .popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 400px;
    height: 300px;

    background-color: rgba(255,255,255,0.1);
    margin: 30px;
    border-radius: 15px;
    /* 阴影 */
    box-shadow: 20px 20px 50px rgba(0,0,0,0.5);
    /* 溢出滚动 */
    overflow: auto;
    display: flex; 
    flex-direction: column; 
    align-items: flex-start; 
    justify-content: center; 
    border-top: 1px solid rgba(255,255,255,0.5);
    border-left: 1px solid rgba(255,255,255,0.5);
    /* 背景模糊 */
    backdrop-filter: blur(10px);
  }
  /* 滚动条设置 */
  .popup::-webkit-scrollbar {
      width: 5px;
      height: 200px;
      background-color: #201d2a;
  }
  .popup::-webkit-scrollbar-track {
      -webkit-box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.3);
      border-radius: 1px;
      background-color: #201d2a;
  }
  .popup::-webkit-scrollbar-thumb {
      border-radius: 10px;
      background-color: rgba(255, 255, 255, 0.3);
  }
  .content{
    display: inline-flex; 
    flex-direction: column; 
    align-items: flex-start; 
    justify-content: center; 
    font-size: 16px;
    padding: 5px 30px 0 30px;
    color: #fff;
    margin: 25px 10px 0 10px;
  }
  .popup input[type=text]{
    width: 130%;
    height: 12%;
    padding: 15px 20px ;
    margin: 5px 10px 10px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
  }

  .popup .close-button{
    display: inline-block;
    width: 76%;
    height: 13%;
    background-color: rgba(71, 157, 255, 0.753);
    color: white;
    padding: 10px 0 10px 0;
    margin: 0px 0px 30px 50px;
    border: none;
    border-radius: 4px;
  }
</style>
  