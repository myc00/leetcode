package dp.hard;

import org.junit.Test;
import tree.TreeNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Program {
    /**
     * 堆箱子
     * @param box
     * @return
     */
    public int pileBox(int[][] box) {
        Arrays.sort(box, (a, b)->a[0] - b[0]);
        int len = box.length;
        int[] dp = new int[len];
        dp[0] = box[0][2];
        int max = dp[0];
        for (int i = 1; i < len; i++) {
            dp[i] = box[i][2];
            for (int j = 0; j < i; j++) {
                if (box[j][0] < box[i][0] &&  box[j][1] < box[i][1] && box[j][2] < box[i][2]){
                    dp[i] = Math.max(dp[i], box[i][2] + dp[j]);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    public List<List<Integer>> BSTSequences(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        LinkedList<Integer> path = new LinkedList<>();
        List<TreeNode> queue = new LinkedList<>();
        if (root == null){
            res.add(path);
            return res;
        }
        queue.add(root);
        BSTSequencesBFS(res, queue, path);
        return res;
    }

    public void BSTSequencesBFS(List<List<Integer>> res, List<TreeNode> queue, LinkedList<Integer> path){
        if (queue.isEmpty()){
            res.add(new ArrayList<>(path));
            return;
        }
        //记录当前的状态，用于回溯
        List<TreeNode> copy = new ArrayList(queue);
        for (int i = 0; i < queue.size(); i++) {
            TreeNode cur = queue.get(i);
            path.offer(cur.val);
            queue.remove(i);
            if (cur.left != null){
                queue.add(cur.left);
            }
            if (cur.right != null){
                queue.add(cur.right);
            }
            BSTSequencesBFS(res, queue, path);
            //进行回溯
            path.pollLast();
            queue = new ArrayList<>(copy);
        }
    }

    /**
     * 直方图的水量
     * @param height
     * @return
     */
    //method1:维护两个数组分别记录当前位置的左右两侧的最大值，取两者的最小值减去当前位置即为水量
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0){
            return 0;
        }
        int[] leftMax = new int[n]; //每个柱子左侧的最大值(包括当前柱子)
        int[] rightMax = new int[n]; //每个柱子右侧的最大值(包括当前柱子)
        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }

        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i > -1; i--) {
            rightMax[i] = Math.max(height[i], rightMax[i + 1]);
        }

        int res = 0;
        for (int i = 0; i < n; i++) {
            res += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return res;
    }
    //method2:空间优化，双指针
    public int trap2(int[] height){
        if (height.length == 0){
            return 0;
        }
        int left = 0, right = height.length - 1, leftMax = 0, rightMax = 0;
        int res = 0;
        while (left < right){
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);

            if (leftMax < rightMax){
                res += leftMax - height[left];
                left++;
            }else {
                res += rightMax - height[right];
                right--;
            }
        }
        return res;
    }

    /**
     * 计算数字1出现的次数
     * @param n
     * @return
     */
    public int countDigitOne(int n) {
        int mulk = 1, res = 0;
        while (n >= mulk){
            res += (n / (mulk * 10)) * mulk + Math.min(Math.max(n % (mulk * 10) - mulk + 1, 0), mulk);
            mulk *= 10;
        }
        return res;
    }

    /**
     * 数字2出现的次数
     * 思路：枚举每一位上2出现的次数
     * @param n
     * @return
     */
    public int numberOf2sInRange(int n) {
        int mulk = 1, res = 0;
        while (n >= mulk){
            res += (n / (mulk * 10)) * mulk + Math.min(Math.max(n % (mulk * 10) - 2 * mulk + 1, 0), mulk);
            mulk *= 10;
        }
        return res;
    }

    /**
     * 最大子数组的和
     * 思路：只要前面的和大于0，那么对于最后求出的最大结果就是有意义的，所以加上当前值
     *      否则，sum重置为当前值
     * @param nums
     */
    public int maxSubArray(int[] nums){
        int max = nums[0], sum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            sum = nums[i] + (sum > 0 ? sum : 0);
            max = Math.max(max, sum);
        }
        return max;
    }

    /**
     * 最大子数组的和,返回左右边界
     * @param nums
     * @return
     */
    public int[] maxSubArray2(int[] nums){
        int max = nums[0], sum = nums[0], start = 0, left = 0, right = 0;
        for (int i = 1; i < nums.length; i++) {
            if (sum > 0){
                sum += nums[i];
            }else {
                sum = nums[i];
                start = i;
            }

            if (sum > max){
                max = sum;
                left = start;
                right = i;
            }
        }
        return new int[]{left, right};
    }

    /**
     * 最大子矩阵
     * 思路：将同一列的行进行合并，随后就变为一维的最大子数组问题，这里用到了前缀和进行优化
     * @param matrix
     * @return
     */
    public int[] getMaxMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] preSum = new int[m + 1][n];
        int max = matrix[0][0];
        //生成前缀和
        for (int i = 1; i <= m; i++) {
            for (int j = 0; j < n; j++) {
                preSum[i][j] = preSum[i - 1][j] + matrix[i - 1][j];
            }
        }
        int[] res = new int[4];
        //合并行
        for (int top = 0; top < m; top++) {
            for (int bottom = top; bottom < m; bottom++) {

                //得到当前合并行所有列的值
                int[] arr = new int[n];
                for (int j = 0; j < n; j++) {
                    arr[j] = preSum[bottom + 1][j] - preSum[top][j];
                }

                //求最大子数组问题
                int sum = arr[0], start = 0;
                for (int i = 1; i < n; i++) {
                    if (sum > 0){
                        sum += arr[i];
                    }else {
                        sum = arr[i];
                        start = i;
                    }

                    if (sum > max){
                        max = sum;
                        res[0] = top;
                        res[1] = start;
                        res[2] = bottom;
                        res[3] = i;
                    }
                }
            }
        }
        return res;
    }

    /**
     * 正则表达式匹配
     * @param s
     * @param p
     * @return
     */
    //method1：dp
    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2]; //*匹配失败或者匹配0次
                    if (match(i, j - 1, s, p)) {  //匹配成功末尾的一个字符并将匹配的该字符丢掉
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    if (match(i, j, s, p)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }
    public boolean match(int i, int j, String s, String p){
        //这个条件需要先进行判断，否则有可能返回true，导致后面出现数组越界
        if (i == 0){
            return false;
        }
        if (p.charAt(j - 1) == '.'){
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    //method2：递归
    public boolean isMatch2(String s, String p) {
        if (p.isEmpty()){
            return s.isEmpty();
        }
        boolean firstMatch = !s.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');

        //p="a*。。。"这种情况
        if (p.length() > 1 && p.charAt(1) == '*'){
            //isMatch(s.substring(1), p)表示*前的字符重复1次以上
            //isMatch(s, p.substring(2))表示重复0次
            return (firstMatch && isMatch2(s.substring(1), p)) || isMatch2(s, p.substring(2));
        }else {
            return firstMatch && isMatch2(s.substring(1), p.substring(1));
        }
    }

    public boolean isMatch3(String s, String p) {
        return isMatchDFS(s, p, 0, 0);
    }
    public boolean isMatchDFS(String s, String p, int i, int j){
        if (j == p.length()){
            return  i == s.length();
        }
        boolean firstMatch = !(i == s.length()) && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.');

        //p="a*。。。"这种情况
        if ((j + 1) < p.length() && p.charAt(j + 1) == '*'){
            //isMatch(s.substring(1), p)表示*前的字符重复1次以上
            //isMatch(s, p.substring(2))表示重复0次
            return (firstMatch && isMatchDFS(s, p, i + 1, j)) || isMatchDFS(s, p, i, j + 2);
        }else {
            return firstMatch && isMatchDFS(s, p, i + 1, j + 1);
        }
    }
    @Test
    public void test(){

    }
}
