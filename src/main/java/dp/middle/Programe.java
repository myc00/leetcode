package dp.middle;

import org.junit.Test;

import java.util.*;

public class Programe {

    /**
     * 最长回文子串
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        int len = s.length();
        if (len == 1){
            return s;
        }
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] chars = s.toCharArray();
        int begin = 0, maxLen = 1;
        for (int l = 2; l <= len; l++){
            //枚举左边界
            for (int i = 0; i < len; i++) {
                int j = l + i - 1; //右边界
                if (j >= len){ //越界
                    break;
                }
                if (chars[i] != chars[j]){
                    dp[i][j] = false;
                }else {
                    if (j - i < 3){ //边界情况(长度为2)
                        dp[i][j] = true;
                    }else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                if (dp[i][j] == true && j - i + 1 > maxLen){
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    /**
     * 生成括号
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<>();
        //method1
//        StringBuffer sb = new StringBuffer();
//        int l = 0, r = 0;
//        dfs(list, sb, l, r, n);

        //method2
//        dfs(list, "", n, 0, 0);

        //method3:3 / 8 个通过测试用例
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < n; i++) {
            sb.append("(");
        }
        for (int i = 0; i < n; i++) {
            sb.append(")");
        }
        list.add(sb.toString());

        for (int i = n - 1; i > 0; i--){
            for (int j = n; j < 2 * n - 1; j++){
                StringBuffer tmp = new StringBuffer(sb);
                tmp.setCharAt(i, ')');
                tmp.setCharAt(j, '(');
                list.add(tmp.toString());
            }
        }
        return list;
    }
    public void dfs(List<String> list, StringBuffer sb, int l, int r, int n){
        if (sb.length() == 2 * n){
            list.add(sb.toString());
            return;
        }
        if (l < n){
            sb.append("(");
            dfs(list, sb, l + 1, r, n);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (r < l){
            sb.append(")");
            dfs(list, sb, l, r + 1, n);
            sb.deleteCharAt(sb.length() - 1);
        }
    }
    public void dfs(List<String> list, String path, int n, int l, int r){
        if (l > n || r > l){
            return;
        }
        if (path.length() == 2 * n){
            list.add(path);
            return;
        }
        dfs(list, path + "(", n, l + 1, r);
        dfs(list, path + ")", n, l, r + 1);
    }

    /**
     * 硬币：背包问题
     * @param n
     * @return
     */
    public int waysToChange(int n) {
        //method1:dfs，超出时间限制
        int[] count = new int[]{0};
        int[] pack = new int[]{1, 5, 10, 25};
        dfs(n, count, pack, 0);
        return count[0];
    }

    //method1: 树的深度遍历，但是单纯的遍历会导致出现重复的结果组合，比如1，5和5，1.所以这里要传递一个index，每次遍历只能枚举当前索引后面的值
                //超出时间限制
    public void dfs(int n, int[] count, int[] pack, int index){
        if (n == 0){
            count[0] = count[0] % 1000000007 + 1;
            return;
        }
        if (n < 0){
            return;
        }
        for (int i = index; i < pack.length; i++){
            dfs(n - pack[i], count, pack, i);
        }
    }

    //method2:动态规划：未优化
    public int waysToChange2(int n){
        int[] coins = new int[]{1, 5, 10, 25};
        int[][] dp = new int[4][n + 1];
        for (int i = 0; i < 4; i++) {
            dp[i][1] = 1;
        }
        for (int i = 1; i <= n; i++){
            dp[0][i] = 1;
        }
        for (int i = 1; i < 4; i++) {
            for (int v = 2; v <= n; v++) {
                for (int j = 0; j <= v / coins[i]; j++){
                    dp[i][v] += dp[i - 1][v - j * coins[i]] % 1000000007;
                }
            }
        }
        return dp[3][n];
    }

    //method3：动态规划+时间优化
    public int waysToChange3(int n){
        int[] coins = new int[]{1, 5, 10, 25};
        int[][] dp = new int[4][n + 1];
        for (int i = 0; i < 4; i++) {
            dp[i][1] = 1;
            dp[i][0] = 1;
        }
        for (int i = 1; i <= n; i++){
            dp[0][i] = 1;
        }
        for (int i = 1; i < 4; i++) {
            for (int v = 2; v <= n; v++) {
                //时间优化
                if (v >= coins[i]){
                    dp[i][v] = (dp[i - 1][v] + dp[i][v - coins[i]]) % 1000000007;
                }else {
                    dp[i][v] = dp[i - 1][v] % 1000000007;
                }
            }
        }
        return dp[3][n];
    }

    //method4：动态规划+时间优化+空间优化
    public int waysToChange4(int n){
        int[] coins = new int[]{1, 5, 10, 25};
        int[] pre = new int[n + 1];
        int[] dp = new int[n + 1];
        int[] swap;
        for (int v = 0; v <= n; v++) {
            pre[v] = 1;
        }
        for (int i = 1; i < 4; i++) {
            dp[0] = 1;
            for (int v = 2; v <= n; v++) {
                if (v >= coins[i]){
                    dp[v] = (pre[v] + dp[v - coins[i]]) % 1000000007;
                }else {
                    dp[v] = pre[v] % 1000000007;
                }
            }
            swap = pre;
            pre = dp;
            dp = swap;
        }
        return dp[n];
    }

    /**
     * 迷路的机器人，寻找可行路径
     * @param obstacleGrid
     * @return
     */
    LinkedList<List<Integer>> res;
    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {
        res = new LinkedList<>();
        LinkedList<List<Integer>> deque = new LinkedList<>();
        int lenX = obstacleGrid.length;
        int lenY = obstacleGrid[0].length;
        pathWithObstaclesDFS(obstacleGrid, deque, 0, 0, lenX, lenY);
        if (!deque.isEmpty()){
            if (deque.peekLast().get(0) != lenX - 1 || deque.peekLast().get(1) != lenY - 1){
                return new LinkedList<>();
            }
        }
        return res;
    }

    //method1：递归，超出时间限制
    public void dfs(int[][] obstacleGrid, Deque<List<Integer>> deque, int x, int y, int lenX, int lenY){
        if (x >= lenX || y >= lenY){
            return;
        }
        if (obstacleGrid[x][y] == 1){
            return;
        }
        if (!deque.isEmpty()){
            if (deque.peekLast().get(0) == lenX - 1 && deque.peekLast().get(1) == lenY - 1){
                return;
            }
        }
        List<Integer> list = new ArrayList<>();
        list.add(x);
        list.add(y);
        deque.offer(list);
        dfs(obstacleGrid, deque, x + 1, y, lenX, lenY);
        dfs(obstacleGrid, deque, x , y + 1, lenX, lenY);
        //当向右向下都走不通时进行回溯，弹出栈顶元素/队列末尾元素，但是只能在未到达终点时弹栈，
        //如果不加终点位置判断条件，最后递归回溯会将路径全部弹出
        if (!deque.isEmpty()){
            if (deque.peekLast().get(0) != (lenX - 1) || deque.peekLast().get(1) != (lenY - 1)){
                deque.pollLast();
            }
        }
    }

    //对遍历过的进行标记
    public void pathWithObstaclesDFS(int[][] obstacleGrid, Deque<List<Integer>> deque, int x, int y, int lenX, int lenY){
        if (x == lenX || y == lenY || x < 0 || y < 0){
            return;
        }
        if (obstacleGrid[x][y] == 1){
            return;
        }
        deque = new LinkedList<>(deque);
        deque.offer(Arrays.asList(x, y));
        obstacleGrid[x][y] = 1;
        if (x == lenX - 1 && y == lenY - 1){
            res = new LinkedList<>(deque);
            return;
        }
        pathWithObstaclesDFS(obstacleGrid, deque, x + 1, y, lenX, lenY);
        pathWithObstaclesDFS(obstacleGrid, deque, x , y + 1, lenX, lenY);
        deque.pollLast();
    }

    /**
     * 马戏团人塔（叠罗汉）
     * @param height
     * @param weight
     * @return
     */
    //method1:动态规划，超时
    public int bestSeqAtIndex(int[] height, int[] weight) {
        int max = 1;
        List<List<Integer>> lists = new ArrayList<>();
        for (int i = 0; i < height.length; i++) {
            lists.add(Arrays.asList(height[i], weight[i]));
        }
        lists.sort(new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                return o1.get(0) - o2.get(0);
            }
        });
        for (List<Integer> list : lists) {
            System.out.print(list.get(0) + ", " + list.get(1) + "\t");
        }
        System.out.println();
        //这里相同身高的人体重降序排列的原因是
        //如果体重升序排序，那么相同身高不同体重的也会在递增子序列里，但是显然相同身高不符合题意
        lists.sort(new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                if (o2.get(0).equals(o1.get(0))){
                    return o2.get(1) - o1.get(1);
                }
                return 0;
            }
        });
        for (List<Integer> list : lists) {
            System.out.print(list.get(0) + ", " + list.get(1) + "\t");
        }
        //查找体重的最长递增子序列
        int[] dp = new int[lists.size()];
        dp[0] = 1;
        for (int i = 1; i < lists.size(); i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (lists.get(j).get(1) < lists.get(i).get(1)){
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            max = Math.max(dp[i], max);
        }
        return max;
    }
    //method2:动态规划+二分查找
    public int bestSeqAtIndex2(int[] height, int[] weight) {
        int len = height.length;
        int[][] person = new int[len][2];
        for (int i = 0; i < len; i++) {
            person[i] = new int[]{height[i], weight[i]};
        }
        Arrays.sort(person, (a, b)->a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int[] dp = new int[len + 1];
        dp[1] = person[0][1];
        int res = 1;
        for (int i = 1; i < len; i++) {
            if (person[i][1] > dp[res]){
                dp[++res] = person[i][1];
            }else {
                int l = 1, r = res, pos = 0;
                while (l <= r){
                    int mid = (l + r) / 2;
                    if (dp[mid] >= person[i][1]){
                        r = mid - 1;
                    }else {
                        l = mid + 1;
                        pos = mid;
                    }
                }
                dp[pos + 1] = person[i][1];
            }
        }
        return res;
    }
    //method3:动态规划+二分查找（调用api）
    public int bestSeqAtIndex3(int[] height, int[] weight) {
        int len = height.length;
        int[][] person = new int[len][2];
        for (int i = 0; i < len; i++) {
            person[i] = new int[]{height[i], weight[i]};
        }
        Arrays.sort(person, (a, b)->a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int[] dp = new int[len + 1];
        dp[1] = person[0][1];
        int res = 1;
        for (int i = 1; i < len; i++) {
            if (person[i][1] > dp[res]){
                dp[++res] = person[i][1];
            }else {
                int index = Arrays.binarySearch(dp, 1, res, person[i][1]);
                if (index < 0){
                    index = -(index + 1); //详情查看binarySearch源码
                }
                dp[index] = person[i][1];
            }
        }
        return res;
    }
    /**
     * 恢复空格
     * @param dictionary
     * @param sentence
     * @return
     */
    //method1：动态规划
    public int respace(String[] dictionary, String sentence) {
        int m = sentence.length();
        int[] dp = new int[m + 1];
        for (int i = 1; i <= m; i++) {
            for (String str : dictionary) {
                int len = str.length();
                if (i >= len && str.equals(sentence.substring(i - len, i))){
                    dp[i] = Math.max(dp[i - len] + len, dp[i]);
                }else {
                    dp[i] = Math.max(dp[i - 1], dp[i]);
                }
            }
        }
        return m - dp[m];
    }
    //字典树
    class Trie {
        public Trie[] next;
        public boolean isEnd;
        public Trie(){
            next = new Trie[26]; //只包含小写字母
            isEnd = false;
        }
        public void insert(String s){
            Trie cur = this;
            for (int i = 0; i < s.length(); i++) {
                int index = s.charAt(i) - 'a';
                if (cur.next[index] == null){
                    cur.next[index] = new Trie();
                }
                cur = cur.next[index];
            }
            cur.isEnd = true;
        }
        public boolean search(String s){
            Trie cur = this;
            for (int i = 0; i < s.length(); i++) {
                int index = s.charAt(i) - 'a';
                if (cur.next[index] == null){
                    return false;
                }
                cur = cur.next[index];
            }
            return cur.isEnd && cur != null;
        }
    }
    //method2：字典树
    public int respace2(String[] dictionary, String sentence) {
        Trie trie = new Trie();
        for (String s : dictionary) {
            trie.insert(s);
        }
        int m = sentence.length();
        int[] dp = new int[m + 1];
        for (int i = 1; i <= m; i++) {
            for (String str : dictionary) {
                int len = str.length();
                if (i >= len && trie.search(sentence.substring(i - len, i))){
                    dp[i] = Math.max(dp[i - len] + len, dp[i]);
                }else {
                    dp[i] = Math.max(dp[i - 1], dp[i]);
                }
            }
        }
        return m - dp[m];
    }

    /**
     * 丑数
     * @param n
     * @return
     */
    public int nthUglyNumber(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        int p2 = 1, p3 = 1, p5 = 1;
        for (int i = 2; i <= n; i++) {
            int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
            dp[i] = Math.min(Math.min(num2, num3), num5);
            if (dp[i] == num2){
                p2++;
            }
            if (dp[i] == num3){
                p3++;
            }
            if (dp[i] == num5){
                p5++;
            }
        }
        return dp[n];
    }

    /**
     * 最大黑方阵
     * @param matrix
     * @return
     */
    public int[] findSquare(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][][] dp = new int[rows][cols][2];

        int r = -1, c = -1, size = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (matrix[row][col] == 0){
                    dp[row][col][0] = 1 + (col > 0 ? dp[row][col - 1][0] : 0);
                    dp[row][col][1] = 1 + (row > 0 ? dp[row - 1][col][1] : 0);

                    for (int side = Math.min(dp[row][col][0], dp[row][col][1]); side > 0; side--){
                        if (dp[row - side + 1][col][0] >= side && dp[row][col - side + 1][1] >= side){
                            if (side > size){
                                r = row - side + 1;
                                c = col - side + 1;
                                size = side;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (r == -1 && c == -1){
            return new int[0];
        }
        return new int[]{r, c, size};
    }

    /**
     * 机器人的运动范围
     * @param m
     * @param n
     * @param k
     * @return
     */
    int[] count = new int[1];
    public int movingCount(int m, int n, int k) {
        int[][] matrix = new int[m][n];
        boolean[][] visited = new boolean[m][n];
        movingCountDFS2(visited, 0, 0, k);
        //method1
//        movingCountDFS(matrix, visited, 0, 0);
        return count[0];
    }

    //method1:递归
    public void movingCountDFS(int[][] arr, boolean[][] visited, int row, int col){
        if (row < 0 || col < 0 || row >= arr.length || col >= arr[0].length){
            return;
        }
        if (visited[row][col] || arr[row][col] == 1){
            return;
        }
        count[0]++;
        visited[row][col] = true;
        movingCountDFS(arr, visited, row + 1, col);
        movingCountDFS(arr, visited, row, col + 1);
        movingCountDFS(arr, visited, row - 1, col);
        movingCountDFS(arr, visited, row, col - 1);
    }

    //method2:递归2
    public int movingCountDFS2(boolean[][] visited, int row, int col, int k){
        if (row < 0 || col < 0 || row >= visited.length || col >= visited[0].length){
            return 0;
        }
        if (visited[row][col]){
            return 0;
        }
        if ((row / 10 + row % 10 + col / 10 + col % 10) > k){
            return 0;
        }
        visited[row][col] = true;
        return movingCountDFS2(visited, row + 1, col, k) + movingCountDFS2(visited, row, col + 1, k)
                + movingCountDFS2(visited, row - 1, col, k) + movingCountDFS2(visited, row, col - 1, k) + 1;
    }

    /**
     * 礼物的最大价值
     * @param grid
     * @return
     */
    public int maxValue(int[][] grid) {
        int rows = grid.length, cols = grid[0].length;
        int[][] dp = new int[rows][cols];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++){
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }

        for (int i = 1; i < cols; i++){
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }

        for (int i = 1; i < rows; i++){
            for (int j = 1; j < cols; j++){
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][cols - 1];
    }

    /**
     * n个骰子的点数:动态规划
     * @param n
     * @return
     */
    public double[] dicesProbability(int n) {
        //dp[i][j]表示i个骰子投出值为j有多少种情况
        //dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j - 2] + ... + dp[i - 1][j - 6]
        int[][] dp = new int[n + 1][n * 6 + 1];
        double[] ans = new double[5 * n + 1];
        double count = Math.pow(6, n);//排列组合,n个骰子的所有情况数
        for (int i = 1; i <= 6; i++) {
            dp[1][i] = 1;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = i; j <= i * 6; j++) {
                for (int k = 1; k <= 6; k++) {
                    dp[i][j] += (j >= k ? dp[i - 1][j - k] : 0);
                }
                if (i == n){
                    ans[j - i] = dp[i][j] /count;
                }

            }
        }
        return ans;
    }

    /**
     * 股票的最大利润
     * 思路:要计算最大利润,那应该是最大值-最小值,且大的在后面
     *     随着数组向后不断遍历,我们只需要维护一个最小值和当前最大利润
     *     当当前数值大于这个最小值时,就计算差值并与我们维护的最大利润作比较,取最大值
     *     当当前数值小于这个最小值时,我们肯定将最小值替换为当前值,因为需要在最低点买入
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int low = 0, max = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] <= prices[low]){
                low = i;
            }
            if (prices[i] > prices[low]){
                max = Math.max(max, prices[i] - prices[low]);
            }
        }
        return max;
    }

    /**
     * 生成括号
     * @param n
     * @return
     */
    public List<String> generateParenthesis2(int n) {
        LinkedList<String> list = new LinkedList<>();
        generateParenthesis2DFS(list, "",  0, 0, n);
        return list;
    }

    public void generateParenthesis2DFS(LinkedList<String> list, String path, int l, int r, int n){
        if (r > n || l > n || r > l){
            return;
        }
        if (path.length() == 2 * n){
            list.offer(path);
            return;
        }
        generateParenthesis2DFS(list, path + "(", l + 1, r, n);
        generateParenthesis2DFS(list, path + ")", l, r + 1, n);
    }

    /**
     * 剪绳子
     * @param n
     * @return
     */
    //method1:动态规划
    public int cuttingRope(int n) {
        if (n == 1){
            return 1;
        }
        if (n == 2){
            return 2;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], dp[j] * (i - j));
                dp[i] = Math.max(dp[i], j * (i - j));
            }
        }
        return dp[n];
    }

    //method2
    public int cuttingRope2(int n) {
        if (n == 1 || n == 2){
            return 1;
        }
        if (n == 3){
            return 2;
        }
        long sum = 1;
        while (n > 4){
            sum *= 3;
            sum %= 1000000007;
            n -= 3;
        }

        return (int)(sum * n % 1000000007);
    }

    /**
     * 回文子串数目
     * 思路：回文串的中心有一个或者两个两种情况。我们枚举每个字符，以当前字符或者当前字符与下一个字符为中心向两侧扩展
     *      如果扩展的左右字符相等，则回文串数量加一
     * @param s
     * @return
     */
    public int countSubstrings(String s) {
        int len = s.length();
        int num = 0;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < 2; j++){ //一个中心或者两个中心
                int l = i, r = i + j;
                while (l >= 0 && r < len && s.charAt(l--) == s.charAt(r++)){
                    num++;
                }
            }
        }
        return num;
    }
    //method2: dp
    public int countSubstrings2(String s) {
        int len = s.length();
        int count = 0;
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
            count++;
        }
        char[] chars = s.toCharArray();
        for (int l = 2; l <= len; l++) {
            for (int i = 0, j = i + l - 1; i < len && j < len; i++, j++) {
                if (l == 2){
                    if (chars[i] == chars[j]){
                        dp[i][j] = true;
                        count++;
                    }
                }else {
                    if (chars[i] == chars[j] && dp[i + 1][j - 1]){
                        dp[i][j] = true;
                        count++;
                    }
                }
            }
        }
        return count;
    }

    public String[][] partition(String s) {
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] chars = s.toCharArray();
        for (int l = 2; l <= len; l++) {
            for (int i = 0, j = i + l - 1; i < len && j < len; i++, j++) {
                if (l == 2){
                    dp[i][j] = chars[i] == chars[j];
                }else {
                    dp[i][j] = chars[i] == chars[j] && dp[i + 1][j - 1];
                }
            }
        }
        List<List<String>> res = new ArrayList<>();
        LinkedList<String> path = new LinkedList<>();
        partitionDFS(res, path, s, dp, len, 0);
        String[][] ans = new String[res.size()][];
        for (int i = 0; i < res.size(); i++) {
            ans[i] = res.get(i).toArray(new String[0]);
        }
        return ans;

    }

    public void partitionDFS(List<List<String>> res, LinkedList<String> path, String s, boolean[][] dp, int len, int pos){
        if (pos == len){
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = pos; i < len; i++) {
            if (dp[pos][i]){
                path.offer(s.substring(pos, i + 1));
                partitionDFS(res, path, s, dp, len, i + 1);
                path.pollLast();
            }
        }
    }

    /**
     * 环形房屋盗窃
     * 思路：两种情况：1）偷第一个房屋但不偷最后一个 2）不偷第一个偷最后一个，最后求max
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int len = nums.length;
        if (len == 1){
            return nums[0];
        }
        int[] dp = new int[len];
        //不偷第一个偷最后一个
        dp[1] = nums[1];
        for (int i = 2; i < len; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        int res = dp[len - 1];

        //偷第一个不偷最后一个
        dp[0] = nums[0];
        dp[1] = dp[0]; //这里不能取前两个的最大值，如果最大值是nums[1]，那么第一个房屋就没有被偷，最后计算的是房屋2~len-1
        for (int i = 2; i < len - 1; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return Math.max(res, dp[len - 2]);
    }

    /**
     * 粉刷房子
     * @param costs
     * @return
     */
    public int minCost(int[][] costs) {
        int len = costs.length;
        if (len == 1){
            return Math.min(costs[0][0], Math.min(costs[0][1], costs[0][2]));
        }
        int[][] dp = new int[len][3];
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];
        for (int i = 1; i < len; i++) {
            dp[i][0] = costs[i][0] + Math.min(dp[i - 1][1], dp[i - 1][2]);
            dp[i][1] = costs[i][1] + Math.min(dp[i - 1][0], dp[i - 1][2]);
            dp[i][2] = costs[i][2] + Math.min(dp[i - 1][0], dp[i - 1][1]);
        }
        return Math.min(dp[len - 1][0], Math.min(dp[len - 1][1], dp[len - 1][2]));
    }

    /**
     * 翻转字符
     * @param s
     * @return
     */
    /*
    method1：dp[i][0]表示前i个字符串以0结尾最少的翻转次数
             dp[i][1]表示前i个字符串以1结尾最少的翻转次数
     */
    public int minFlipsMonoIncr(String s) {
        int len = s.length();
        int[][] dp = new int[len][2];

        dp[0][0] = s.charAt(0) == '0' ? 0 : 1;
        dp[0][1] = s.charAt(0) == '1' ? 0 : 1;
        for (int i = 1; i < len; i++) {
            if (s.charAt(i) == '0'){
                dp[i][0] = dp[i - 1][0];
                dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + 1;
            }else {
                dp[i][0] = dp[i - 1][0] + 1;
                dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]);
            }
        }
        return Math.min(dp[len - 1][0], dp[len - 1][1]);
    }

    //method1空间降维
    public int minFlipsMonoIncr2(String s) {
        int len = s.length();
        int[][] dp = new int[len][2];

        int o0 = s.charAt(0) == '0' ? 0 : 1;
        int o1 = s.charAt(0) == '1' ? 0 : 1;
        int c0 = o0, c1 = o1;
        for (int i = 1; i < len; i++) {
            if (s.charAt(i) == '0'){
                c0 = o0;
                c1 = Math.min(o0, o1) + 1;
            }else {
                c0 = o0 + 1;
                c1 = Math.min(o0, o1);
            }
            o0 = c0;
            o1 = c1;
        }
        return Math.min(c0, c1);
    }

    /**
     * 最长斐波那契数列
     * 思路：数列的最后两个数字能够决定一整个数列的唯一性，不会出现多种情况
     *      令dp[i][j]为以arr[i][j]结尾的斐波那契数列的长度，则dp[i][j] = dp[k][i] + 1（k < i < j且arr[k]+arr[i]=arr[j]）
     * @param arr
     * @return
     */
    public int lenLongestFibSubseq(int[] arr) {
        int len = arr.length, max = 0;
        int[][] dp = new int[len][len];
        for (int i = 2; i < len; i++) {
            //双指针枚举小于i的所有情况
            int l = 0, r = i - 1;
            while (l < r){
                int sum = arr[l] + arr[r];
                if (sum == arr[i]){
                    dp[r][i] = dp[l][r] + 1;
                    max = Math.max(max, dp[r][i] + 2);
                    l++;
                    r--;
                }else if (sum < arr[i]){
                    l++;
                }else {
                    r--;
                }
            }
        }
        return max;
    }

    /**
     * 最长公共子序列：典型动态规划
     * @param text1
     * @param text2
     * @return
     */
    public int longestCommonSubsequence(String text1, String text2) {
        int n1 = text1.length(), n2 = text2.length();
        char[] chars1 = text1.toCharArray();
        char[] chars2 = text2.toCharArray();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 1; i < n1; i++) {
            for (int j = 1; j < n2; j++) {
                if (chars1[i - 1] == chars2[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n1][n2];
    }

    /**
     * 字符串交织
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != (s1.length() + s2.length())){
            return false;
        }
        return isInterleaveDFS(s1, s2, s3, 0, 0, 0);
    }
    //method1:dfs
    public boolean isInterleaveDFS(String s1, String s2, String s3, int index1, int index2, int index3){
        if (index3 == s3.length()){
            return true;
        }
        boolean res = false;
        if (index1 < s1.length() && s1.charAt(index1) == s3.charAt(index3)){
            res |= isInterleaveDFS(s1, s2, s3, index1 + 1, index2, index3 + 1);
        }
        if (index2 < s2.length() && s2.charAt(index2) == s3.charAt(index3)){
            res |= isInterleaveDFS(s1, s2, s3, index1, index2 + 1, index3 + 1);
        }
        return res;
    }
    //method2:dp
    public boolean isInterleave2(String s1, String s2, String s3) {
        if (s3.length() != (s1.length() + s2.length())){
            return false;
        }
        int n1 = s1.length(), n2 = s2.length(), n3 = s3.length();
        boolean[][] dp = new boolean[n1 + 1][n2 + 1];
        dp[0][0] = true;
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        char[] c3 = s3.toCharArray();
        for (int i = 1; i <= n1; i++) {
            if (c1[i - 1] != c3[i - 1]){
                break;
            }
            dp[i][0] = true;
        }
        for (int i = 1; i <= n2; i++) {
            if (c2[i - 1] != c3[i - 1]){
                break;
            }
            dp[0][i] = true;
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                dp[i][j] = (dp[i - 1][j] && c1[i - 1] == c3[i + j - 1]) || (dp[i][j - 1] && c2[j - 1] == c3[i + j - 1]);
            }
        }
        return dp[n1][n2];
    }

    /**
     * 最小路径和
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        int rows = grid.length, cols = grid[0].length;
        int[][] dp = new int[rows][cols];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < cols; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < rows; i++) {
            for (int j = 1; j < cols; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[rows - 1][cols - 1];
    }

    /**
     * 三角形中最小路径之和
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.size() == 1){
            return triangle.get(0).get(0);
        }
        int n = triangle.size();
        int[][] dp = new int[n][n];
        dp[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < n; i++) {
            dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
        }
        int min = dp[n - 1][0];
        for (int i = 1; i < n; i++) {
            int j = 1;
            for (; j < triangle.get(i).size() - 1; j++) {
                dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
                if (i == n - 1){
                    min = Math.min(min, dp[i][j]);
                }
            }
            if (j == i){
                dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
                if (i == n - 1){
                    min = Math.min(min, dp[i][j]);
                }
            }
        }
        return min;
    }
    //空间优化o(n)
    public int minimumTotal2(List<List<Integer>> triangle) {
        if (triangle.size() == 1){
            return triangle.get(0).get(0);
        }
        int n = triangle.size();
        int[] pre = new int[n];
        int[] cur = new int[n];
        pre[0] = triangle.get(0).get(0);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            int size = triangle.get(i).size();
            cur[0] = pre[0] + triangle.get(i).get(0);
            for (int j = 1; j < size - 1; j++) {
                cur[j] = Math.min(pre[j], pre[j - 1]) + triangle.get(i).get(j);
            }
            cur[i] = pre[i - 1] + triangle.get(i).get(i);
            for (int j = 0; j <= i; j++) {
                pre[j] = cur[j];
            }
            if (i == n - 1){
                for (int j = 0; j < i; j++) {
                    min = Math.min(min, cur[j]);
                }
            }
        }
        return min;
    }
    //原地修改
    public int minimumTotal3(List<List<Integer>> triangle) {
        if (triangle.size() == 1){
            return triangle.get(0).get(0);
        }
        int n = triangle.size();
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            int size = triangle.get(i).size();
            for (int j = 0; j < size; j++) {
                if (j == 0){
                    triangle.get(i).set(j, triangle.get(i - 1).get(0) + triangle.get(i).get(0));
                }else if (j == size - 1){
                    triangle.get(i).set(j, triangle.get(i - 1).get(j - 1) + triangle.get(i).get(j));
                }else {
                    triangle.get(i).set(j, triangle.get(i).get(j) +
                            Math.min(triangle.get(i - 1).get(j - 1), triangle.get(i - 1).get(j)));
                }
            }
        }
        for (int i = 0; i < n; i++) {
            min = Math.min(min, triangle.get(n - 1).get(i));
        }
        return min;
    }

    /**
     * 加减的目标值
     * @param nums
     * @param target
     * @return
     */
    public int findTargetSumWays(int[] nums, int target) {
        int[] count = new int[]{0};
        findTargetSumWaysDFS(nums, target, count, 1, 0);
        findTargetSumWaysDFS(nums, target, count, -1, 0);
        return count[0];
    }
    //method1：递归+回溯
    public void findTargetSumWaysDFS(int[] nums, int target, int[] count, int multiplier, int index){
        if (index == nums.length){
            return;
        }
        if (index < nums.length){
            target += multiplier * nums[index];
        }
        if (index == nums.length - 1 && target == 0){
            count[0]++;
        }
        findTargetSumWaysDFS(nums, target, count, 1, index + 1);
        findTargetSumWaysDFS(nums, target, count, -1, index + 1);
    }
    //method2：dp
    public int findTargetSumWays2(int[] nums, int target) {
        int len = nums.length, sum = 0;
        for (int i = 0; i < len; i++) {
            sum += nums[i];
        }
        int neg = (sum - target);
        if (neg < 0 || neg % 2 != 0){
            return 0;
        }
        neg /= 2;
        //下面就是背包问题,背包容量为neg，物品为nums
        int[][] dp = new int[len + 1][neg + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= len; i++) {
            for (int j = 0; j <= neg; j++) {
                if (nums[i - 1] > j){
                    dp[i][j] = dp[i - 1][j];
                }else {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[len][neg];
    }

    /**
     * 最少的硬币数
     * @param coins
     * @param amount
     * @return
     */
    public int coinChange(int[] coins, int amount) {
        int[] min = new int[]{Integer.MAX_VALUE};
        coinChangeDFS(coins, amount, 0, 0, min);
        if (min[0] == Integer.MAX_VALUE){
            return -1;
        }
        return min[0];
    }
    //method1:递归+回溯
    public void coinChangeDFS(int[] coins, int amount, int count, int index, int[] min){
        if (amount == 0){
            min[0] = Math.min(min[0], count);
            return;
        }
        if (amount < 0){
            return;
        }
        for (int i = index; i < coins.length; i++) {
            coinChangeDFS(coins, amount - coins[i], count + 1, i, min);
        }
    }
    //method2:dp[i]表示组成i面值需要的最少硬币数,dp[i] = min(dp[i - num]) + 1
    public int coinChange2(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (i - coin >= 0){
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    /**
     * 排列的数目
     * @param nums
     * @param target
     * @return
     */
    public int combinationSum4(int[] nums, int target) {
        int[] count = new int[]{0};
        combinationSum4DFS(nums, target, count);
        return count[0];
    }
    //method1: 递归+回溯
    public void combinationSum4DFS(int[] nums, int target, int[] count){
        if (target == 0){
            count[0]++;
            return;
        }
        if (target < 0){
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            combinationSum4DFS(nums, target - nums[i], count);
        }
    }
    //method2: dp[i]表示组成target需要多少种方法，dp[i] = dp[i - num[0]] +...+ dp[i - num[num.length - 1]]
    public int combinationSum42(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (i >= num){
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }

    /**
     * 矩阵中的距离
     * @param mat
     * @return
     */
    //method1：广度搜索
    public int[][] updateMatrix(int[][] mat) {
        //1.先找到所有为0的格子并添加进队列(fifo)
        LinkedList<int[]> queue = new LinkedList<>();
        int rows = mat.length, cols = mat[0].length;
        boolean[][] visited = new boolean[rows][cols]; //防止同一个格子被重复计算
        //值为0的格子作为最内圈，其距离为0
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat[i][j] == 0){
                    queue.offer(new int[]{i, j});
                    visited[i][j] = true;
                }
            }
        }
        int[][] directions = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; //上下左右四个方向
        int[][] res = new int[rows][cols]; //最终返回结果
        int cost = 0; //距离值
        while (!queue.isEmpty()){
            int len = queue.size();
            //每一轮添加进队列的格子具有相同的距离,一个循换内的所有格子具有相同的距离
            for (int l = 0; l < len; l++) {
                int[] grid = queue.poll();

                if (mat[grid[0]][grid[1]] == 1){
                    res[grid[0]][grid[1]] = cost;
                }

                for (int[] direction : directions) {
                    int x = grid[0] + direction[0];
                    int y = grid[1] + direction[1];
                    if (x >= 0 && x < rows && y >= 0 && y < cols && !visited[x][y]){
                        queue.offer(new int[]{x, y});
                        visited[x][y] = true;
                    }
                }
            }
            cost++;//每一轮循环结束，就是同样距离的一圈结束
        }
        return res;
    }

    //method2：双向dp
    public int[][] updateMatrix2(int[][] mat){
        /*
        分别从左上、右下两个方向进行dp，并取最小值即可
        左上到右下的dp可以得到距离左侧和上侧的0最小的距离
        右下到左上的dp可以得到距离右侧和下侧的0最小的距离
        两个方向取最小值即可
         */
        int rows = mat.length, cols = mat[0].length, MAX = (int)1e9;
        int[][] dp = new int[rows][cols];
        for (int i = 0; i < dp.length; i++) {
            Arrays.fill(dp[i], MAX);
        }

        for (int i = 0; i < rows; i--) {
            for (int j = 0; j < cols; j--) {
                if (mat[i][j] == 0){
                    dp[i][j] = 0;
                }
            }
        }

        //从左上往右下dp
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dp[i][j] = Math.min(dp[i][j], i - 1 >= 0 ? dp[i - 1][j] + 1 : MAX);
                dp[i][j] = Math.min(dp[i][j], j - 1 >= 0 ? dp[i][j - 1] + 1 : MAX);
            }
        }

        //从右下往左上dp
        for (int i = rows - 1; i > -1; i++) {
            for (int j = cols - 1; j > -1; j++) {
                dp[i][j] = Math.min(dp[i][j], i + 1 < rows ? dp[i + 1][j] + 1 : MAX);
                dp[i][j] = Math.min(dp[i][j], j + 1 < cols ? dp[i][j + 1] + 1 : MAX);
            }
        }
        return dp;
    }

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 把数字翻译成字符串
     * @param num
     * @return
     */
    public int translateNum(int num) {
        if (num < 10){
            return 1;
        }
        List<Integer> list = new ArrayList<>();
        while (num > 0){
            list.add(num % 10);
            num /= 10;
        }
        Collections.reverse(list);
        int len = list.size();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= len; i++) {
            dp[i] = dp[i - 1];
            int sum = list.get(i - 2) * 10 + list.get(i - 1);
            if (list.get(i - 2) != 0 && sum < 26){
                dp[i] += dp[i - 2];
            }
        }
        return dp[len];
    }

    /**
     * 最长递增子序列
     * @param nums
     * @return
     */
    //动态规划+二分法
    public int lengthOfLIS(int[] nums) {
        //len是最终子序列的长度
        int len = 1, n = nums.length;
        int[] dp = new int[n + 1];
        dp[len] = nums[0];
        for (int i = 1; i < n; i++) {
            if (nums[i] > dp[len]){
                dp[++len] = nums[i];
            }else {
                //二分查找dp序列中小于nums[i]的最大值，并将该值后面的第一个值替换为nums[i]
                int l = 1, r = len, pos = 0;
                while (l <= r){
                    int mid = (l + r) / 2;
                    if (dp[mid] >= nums[i]){
                        r = mid - 1;
                    }else {
                        pos = mid;
                        l = mid + 1;
                    }
                }
                dp[pos + 1] = nums[i];
            }
        }
        return len;
    }

    @Test
    public void test(){
        System.out.println(translateNum(1068385902));
    }
}
