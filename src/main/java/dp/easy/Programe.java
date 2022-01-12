package dp.easy;

import org.junit.Test;
import org.omg.CORBA.INTERNAL;
import tree.ListNode;
import tree.TreeNode;

import java.util.*;

public class Programe {

    public int rob(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < nums.length; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[nums.length - 1];
    }

    public int rob2(int[] nums) {
        int dp_l = nums[0], dp_r = Math.max(nums[0], nums[1]);

        for (int i = 2; i <nums.length; i++){
            int dp_i = Math.max(dp_l + nums[i], dp_r);
            dp_l = dp_r;
            dp_r = dp_i;
        }
        return dp_r;
    }

    /**
     * 杨辉三角
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows == 1){
            res.add(Arrays.asList(1));
            return res;
        }

        List<Integer> list1 = Arrays.asList(1);
        res.add(list1);
        for (int i = 1; i < numRows; i++){
            List<Integer> list = res.get(i - 1);
            List<Integer> cur = new ArrayList<>();
            cur.add(1);
            for (int j = 0; j < i - 1; j++){
                cur.add(list.get(j) + list.get(j + 1));
            }
            cur.add(1);
            res.add(cur);
        }
        return res;
    }

    /**
     * 杨辉三角Ⅱ
     * @param rowIndex
     * @return
     */
    public List<Integer> generate2(int rowIndex) {
        List<Integer> pre = Arrays.asList(1);
        List<Integer> cur = pre;
        for (int i = 1; i <= rowIndex; i++){
            cur = new ArrayList<>();
            cur.add(1);
            for (int j = 0; j < pre.size() - 1; j++){
                cur.add(pre.get(j) + pre.get(j + 1));
            }
            cur.add(1);
            pre = cur;
        }
        return cur;
    }

    /**
     * 获取生成数组中的最大值
     * @param n
     * @return
     */
    public int getMaximumGenerated(int n) {
        if (n == 0){
            return 0;
        }
        int[] nums = new int[n + 1];
        nums[0] = 0;
        nums[1] = 1;
        int count = n / 2, max = 1;
        for (int i = 1; i <= count; i++) {
            nums[2 * i] = nums[i];
            max = Math.max(max, nums[2 * i]);
            if ((2 * i + 1) <= n){
                nums[2 * i + 1] = nums[i] + nums[i + 1];
                max = Math.max(max, nums[2 * i + 1]);
            }

        }
        return max;
    }

    public int fib(int n) {
        if (n == 1){
            return n;
        }
        return fib(n - 1) + fib(n - 2);
    }

    /**
     * 小孩上楼梯:dp[n]表示到第n层楼梯需要几种方法
     *          思路：到达第n层楼梯，走最后一步有三种选择：
     *                 1）走一步 2）走两步 3）走三步，分别对应dp[n - 1], dp[n - 2], dp[n - 3]
     * @param n
     * @return
     */
    public int waysToStep(int n) {
        if (n == 0 || n == 1){
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        if (n ==2) {
            return 2;
        }
        dp[2] = 2;
        for (int i = 3; i <= n; i++){
            dp[i] = dp[i - 2] + dp[i - 3];
            dp[i] %= 1000000007;
            dp[i] += dp[i - 1];
            dp[i] %= 1000000007;
        }
        return dp[n];
    }

    /**
     * 翻转数位：给定一个32位整数 num，你可以将一个数位从0变为1。请编写一个程序，找出你能够获得的最长的一串1的长度。
     *      思路：关键点在于当遇到两个及以上的连续0要进行计数重置。这里定义两个变量cur、insert
     *          当不为0时，两个变量都加1，当为0时，填充1，insert = cur + 1，cur则重置为0，并记录当前的计数最大值
     * @param num
     * @return
     */
    public int reverseBits(int num) {
        int res = 1, cur = 0, insert = 0;
        for (int i = 0; i < 32; i++){
            if ((num & (1 << i)) != 0){ //判断当前位是不是0
                cur ++;
                insert ++;
            }else {
                insert = cur + 1;
                cur = 0;
            }
            res = Math.max(res, insert);
        }
        return res;
    }

    /**
     * 除数博弈: 先手是否成功
     * @param n
     * @return
     */
    public boolean divisorGame(int n) {
        if (n == 1){
            return false;
        }
        boolean[] f = new boolean[n + 1]; //先手i是否成功
        f[1] = false;
        f[2] = true;
        for (int i = 3; i <= n; i++){
            for (int j = 1; j < i; j++){
                if ((i % j) == 0 && !f[i - j]){ //如果j是i的因数并且后手失败
                    f[i] = true;
                    break;
                }
            }
        }
        return f[n];
    }

    /**
     * 泰波那契序列 Tn 定义如下： 
     *  T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2
     * 给你整数 n，请返回第 n 个泰波那契数 Tn 的值。
     * @param n
     * @return
     */
    public int tribonacci(int n) {
        if (n == 0 || n == 1){
            return n;
        }
        if (n == 2){
            return 1;
        }
        int t0 = 0, t1 = 1, t2 = 1, t3 = 0;
        for (int i = 3; i <= n; i++){
            t3 = t0 + t1 + t2;
            t0 = t1;
            t1 = t2;
            t2 = t3;
        }
        return t3;
    }

    /**
     * 传递信息
     * @param n
     * @param relation
     * @param k
     * @return
     */
    public int numWays(int n, int[][] relation, int k) {
        int count = 0;
        List<List<Integer>> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            edges.add(new ArrayList<>());
        }
        for (int[] edge : relation) {
            edges.get(edge[0]).add(edge[1]);
        }
        Deque<Integer> deque = new LinkedList<>();
        deque.offer(0);

        while (!deque.isEmpty() && k > 0){
            k--;
            int size = deque.size();
            for (int i = 0; i < size; i++){
                int first = deque.poll();
                List<Integer> list = edges.get(first);
                for (int dist : list) {
                    deque.offer(dist);
                }
            }

        }
        if (k == 0){
            while (!deque.isEmpty()){
                if (deque.poll() == n-1){
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * 连续数列最大值
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
//        int[] dp = new int[nums.length];
//        dp[0] = nums[0];
//        int max = nums[0];
//        for (int i = 1; i < nums.length; i++) {
//            dp[i] = Math.max(nums[i], dp[i - 1] + nums[i]);
//            max = Math.max(max, dp[i]);
//        }
//        return max;

        int max = nums[0], pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            pre = Math.max(pre + nums[i], nums[i]);
            max = Math.max(pre, max);
        }
        return max;
    }

    /**
     * 按摩师
     * @param nums
     * @return
     */
    public int massage(int[] nums) {
        int n = nums.length;
        if (n == 0){
            return 0;
        }
        if (n == 1){
            return nums[0];
        }
        if (n == 2){
            return Math.max(nums[0], nums[1]);
        }
//        int[] dp = new int[n];
//        dp[0] = nums[0];
//        dp[1] = Math.max(dp[0], nums[1]);
//        for (int i = 2; i < n; i++) {
//            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
//        }
//        return dp[n - 1];

        int order1 = nums[0], order2 = Math.max(nums[0], nums[1]), max = 0;
        for (int i = 2; i < n; i++) {
            max = Math.max(order2, order1 + nums[i]);
            order1 = order2;
            order2 = max;
        }
        return max;
    }

    /**
     * 前n个数字二进制中1的个数（1比特数）
     *  思路：最高有效位（小于等于i的最大的2的整数次幂数），i的1比特数 = （i - 最高有效位）的一比特数加1
     *         最高有效位：x & (x - 1) == 0
     * @param n
     * @return
     */
    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 0;
        int highBits = 0; //最高有效位
        for (int i = 1; i <= n; i++) {
            if ((i & (i - 1)) == 0){
                highBits = i;
            }
            dp[i] = dp[i - highBits] + 1;
        }
        return dp;
    }

    /**
     * 爬楼梯的最少成本：一定要加上当前的值，否则会导致计算的最少成本跨度大于2
     * @param cost
     * @return
     */
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
//        int[] dp = new int[n];
//        dp[0] = cost[0];
//        dp[1] = cost[1];
//        for (int i = 2; i < n; i++){
//            dp[i] = Math.min(dp[i - 2], dp[i - 1]) + cost[i];
//        }
//        return Math.min(dp[n - 2], dp[n - 1]);
        int o1 = cost[0], o2 = cost[1], res = 0;
        for (int i = 2; i < n; i++) {
            res = Math.min(o1, o2) + cost[i];
            o1 = o2;
            o2 = res;
        }
        return Math.min(o1, o2);
    }

    @Test
    public void test(){
        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
    }
}
