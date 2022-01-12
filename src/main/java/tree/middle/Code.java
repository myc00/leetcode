package tree.middle;


import com.sun.org.apache.bcel.internal.generic.LNEG;
import com.sun.org.apache.bcel.internal.generic.LSHL;
import org.junit.Test;
import sun.awt.image.ImageWatched;
import tree.ListNode;
import tree.TreeNode;

import java.util.*;

public class Code {

    static boolean flag[] = new boolean[]{true};
    static long[] former = new long[]{Long.MIN_VALUE};


    //检验是否为二叉搜索树
    public static boolean isValidBST(TreeNode node, long[] former, boolean[] flag) {

        if (node == null){
            return true;
        }
        isValidBST(node.left, former, flag);
        if (node.val <= former[0]){
            flag[0] = false;
            return flag[0];
        }
        former[0] = node.val;
        isValidBST(node.right, former, flag);
        return flag[0];
    }

    //给定一个二叉树，返回其节点值的锯齿形层序遍历(即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行)
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null){
            return new ArrayList<List<Integer>>();
        }

        Deque<TreeNode> deque1 = new LinkedList<>();
        Deque<TreeNode> deque2 = new LinkedList<>();
        deque1.offer(root);
        List<List<Integer>> outerList = new ArrayList<>();
        List<Integer> innerList = new ArrayList<>();
        while (!deque1.isEmpty() || !deque2.isEmpty()){
            while (!deque1.isEmpty()){

                TreeNode node = deque1.pollFirst();
                innerList.add(node.val);
                if (node.left != null){
                    deque2.offerLast(node.left);
                }
                if (node.right != null){
                    deque2.offerLast(node.right);
                }
            }
            if (!innerList.isEmpty()){
                outerList.add(new ArrayList<>(innerList));
                innerList.clear();
            }

            while (!deque2.isEmpty()){
                //反向读取
                TreeNode node = deque2.pollLast();
                innerList.add(node.val);
                if (node.right != null){
                    deque1.offerFirst(node.right);
                }
                if (node.left != null){
                    deque1.offerFirst(node.left);
                }
            }
            if (!innerList.isEmpty()){
                outerList.add(new ArrayList<>(innerList));
                innerList.clear();
            }
        }
        return outerList;
    }

    //根据前序遍历以及中序遍历构造二叉树并返回根节点
    public TreeNode buildTree(int[] preorder, int[] inorder) {

        int pLen = preorder.length;
        int inLen = inorder.length;
        if (pLen != inLen){
            return null;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }

        return buildTree(preorder, 0, pLen - 1, map, inorder, 0, inLen - 1);
    }
    public TreeNode buildTree(int[] preorder, int preLeft, int preRight, Map<Integer, Integer> map, int[] inorder, int inLeft, int inRight) {

        if (preLeft > preRight || inLeft > inRight){
            return null;
        }
        int rootVal = preorder[preLeft];
        TreeNode node = new TreeNode(rootVal);
        int pIndex = map.get(rootVal);
        node.left = buildTree(preorder, preLeft + 1, pIndex - inLeft + preLeft, map, inorder, inLeft, pIndex - 1);
        node.right = buildTree(preorder, pIndex - inLeft + preLeft + 1, preRight, map, inorder, pIndex + 1, inRight);

        return node;
    }

    //根据一棵树的中序遍历与后序遍历构造二叉树
    public static TreeNode buildTree2(int[] inorder, int[] postorder) {
        int pLen = postorder.length;
        int inLen = inorder.length;
        if (pLen != inLen){
            return null;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return buildTree2(inorder, 0, inLen - 1, map, postorder, 0, pLen - 1);
    }
    public static TreeNode buildTree2(int[] inorder, int inLeft, int inRight, Map<Integer, Integer> map, int[] postorder, int postLeft, int postRight) {
        if (inLeft > inRight || postLeft > postRight){
            return null;
        }

        int rootVal = postorder[postRight];
        TreeNode node = new TreeNode(rootVal);
        int pIndex = map.get(rootVal);
        node.left = buildTree2(inorder, inLeft, pIndex - 1, map, postorder, postLeft, pIndex - inLeft + postLeft - 1);
        node.right = buildTree2(inorder, pIndex + 1, inRight, map, postorder, pIndex - inLeft + postLeft, postRight - 1);
        return node;
    }

//    给定一个二叉树，返回其节点值自底向上的层序遍历
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null){
            return new ArrayList<List<Integer>>();
        }

        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerFirst(root);

        LinkedList<List<Integer>> result = new LinkedList<>();
        while (!deque.isEmpty()){
            int len = deque.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < len; i++){
                TreeNode node = deque.pollFirst();
                list.add(node.val);
                if (node.left != null){
                    deque.offerLast(node.left);
                }
                if (node.right != null){
                    deque.offerLast(node.right);
                }
            }
            result.offerFirst(list);
        }
        return result;
    }



    //给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null){
            return null;
        }
        return sortedListToBST(head, null);
    }
    public TreeNode sortedListToBST(ListNode left, ListNode right) {
        if (left == right){
            return null;
        }
        //快慢指针，快速找到中间节点
        ListNode mid = getMid(left, right);
        TreeNode node = new TreeNode(mid.val);
        node.left = sortedListToBST(left, mid);
        node.right = sortedListToBST(mid.next, right);
        return node;
    }
    //快慢指针，快速找到中间节点
    public ListNode getMid(ListNode left, ListNode right) {
        //快慢指针，快速找到中间节点
        ListNode slow = left;
        ListNode fast = left;
        while (fast != right && fast.next != right){
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    //路经总和（找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径）
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root == null){
            return new ArrayList<List<Integer>>();
        }
        LinkedList<Integer> path = new LinkedList<>();
        List<List<Integer>> list = new LinkedList<>();
        pathSum(root, targetSum, path, list);
        return list;
    }
    public void pathSum(TreeNode node, int targetSum, LinkedList<Integer> path, List<List<Integer>> list){
        if (node == null){
            return;
        }
        path.offerLast(node.val);
        if (node.left == null && node.right == null){
            if (node.val == targetSum){
                list.add(new ArrayList<>(path));
            }
        }else {
            pathSum(node.left, targetSum - node.val, path, list);
            pathSum(node.right, targetSum - node.val, path, list);
        }
        path.pollLast();
    }
    //方法2（广度优先搜索）
    List<List<Integer>> list = new ArrayList<>();
    HashMap<TreeNode, TreeNode> map = new HashMap<>();
    public List<List<Integer>> pathSum2(TreeNode root, int targetSum) {
        if (root == null){
            return new ArrayList<List<Integer>>();
        }
        Deque<TreeNode> queNode = new LinkedList<>();
        Deque<Integer> queSum = new LinkedList<>();
        queNode.offerLast(root);
        queSum.offerLast(root.val);

        while (!queNode.isEmpty()){
            TreeNode node = queNode.pollFirst();
            int tmp = queSum.pollFirst();
            if (node.left == null && node.right == null){
                if (tmp == targetSum){
                    getPath(node);
                }
            }
            if (node.left != null){
                map.put(node.left, node);
                queNode.offerLast(node.left);
                queSum.offerLast(tmp + node.left.val);
            }
            if (node.right != null){
                map.put(node.right, node);
                queNode.offerLast(node.right);
                queSum.offerLast(tmp + node.right.val);
            }
        }
        return list;
    }
    public void getPath(TreeNode node){
        List<Integer> path = new ArrayList<>();
        while (node != null){
            path.add(node.val);
            node = map.get(node);
        }
        Collections.reverse(path);
        list.add(new ArrayList<>(path));
    }

    /**
     * 二叉树的右视图
     * 思路1：每次循环遍历的都是右子节点（如果右子节点不为空的话）
     *      pre记录当前节点的父节点，deque维护一个当前节点的左兄弟节点的队列
     *      当前节点不为空时，将当前节点添加到list集合中，并遍历队列中的所有节点并弹出，将它们的所有子节点添加到队列中
     *      直到当前节点为空时，只需要将队列尾部的元素添加到list中，并重复之前的操作，将所有子节点添加到队列
     * 思路2：广度遍历将队列尾部元素添加到list即可
     * 思路3：（根->右->左）的深度遍历来获得最右侧的值，不过可能要维护当前节点的深度
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null){
            return list;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        deque.offerLast(root);
        TreeNode node;
        while (!deque.isEmpty()){
            list.add(deque.peekLast().val);
            int len = deque.size();
            for (int i = 0; i < len; i++) {
                node = deque.pollFirst();
                if (node.left != null){
                    deque.offerLast(node.left);
                }
                if (node.right != null){
                    deque.offerLast(node.right);
                }
            }
        }
        return list;
    }

    /**
     * 计算完全二叉树的节点个数
     * 思路1：暴力遍历
     * 思路2：位运算+二分查找
     *      将根节点记为1，每个节点都有一个按照顺序遍历对应的值。将该值转为二进制，其中0为左子节点，1为右子节点
     *      那么改二进制可以表示从根节点到该节点的路径。我们可以利用该特性
     *      首先深度优先遍历左子树，得到该树的最大深度h（从0开始记），那么该树的节点总数一定在2的h次方到2的h+1次方-1之间
     *      此时我们得到了节点总数的左右边界值。以此边界值进行二分查找，算出每次中间的叶子节点对应的值
     *      将值与上一层的左边界进行循环&运算，每次&运算都将该节点二进制右移一位并且按照运算结果得到node的左/右子节点，当该二进制右移完成如果最后的节点为null，那么
     *      说明该节点为null
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if (root == null){
            return 0;
        }
        int level = 0;
        TreeNode node = root;
        while (node.left != null){
            level += 1;
            node = node.left;
        }
        int left = 1 << level, right = (1 << (level + 1)) - 1;

        while (left < right){
            int mid = left + (right - left + 1) / 2;
            if (exists(root, level, mid)){
                left = mid;
            }else {
                right = mid - 1;
            }
        }
        return left;
    }
    public boolean exists(TreeNode root, int level, int k){
        int bits = 1 << (level - 1); //得到最深层上一层的左边界
        TreeNode node = root;
        while (bits > 0 && node != null){
            if ((bits & k) == 0){
                node = node.left;
            }else {
                node = node.right;
            }
            bits = bits >> 1;
        }
        return node != null;
    }

    /**
     * 倒数第k个最小值（从1开始）
     * 思路1：中序遍历得到递增序列，返回第k个元素
     * @param root
     * @param k
     * @return
     */
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> deque = new LinkedList<>();
        TreeNode node = root;
        while (!deque.isEmpty() || node != null){
            while (node != null){
                deque.offerLast(node);
                node = node.left;
            }
            node = deque.pollLast();
            k--;
            if (k == 0 && node != null){
                return node.val;
            }
            node = node.right;
        }
        return 0;
    }

    /**
     * 找到一个二叉树中指定两个节点p，q的最近公共祖先
     * @param root
     * @param p
     * @param q
     * @return
     */
    TreeNode res;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        postOrder(root, p, q);
        return res;
    }

    public boolean postOrder(TreeNode root, TreeNode p, TreeNode q){
        if (root == null){
            return false;
        }
        boolean lSon = postOrder(root.left, p, q);
        boolean rSon = postOrder(root.right, p, q);
        if ((lSon && rSon) || ((root.val == p.val || root.val == q.val) && (lSon || rSon))){
            res = root;
        }
        return lSon || rSon || (root.val == p.val || root.val == q.val);
    }

    //方法2
    Map<Integer, TreeNode> parent = new HashMap<>();
    List<Integer> visited = new ArrayList<>();
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null){
            return null;
        }
        preOrder(root);
        while (p != null){
            visited.add(p.val);
            p = parent.get(p.val);
        }
        while (q != null){
            if (visited.contains(q.val)){
                return q;
            }
            q = parent.get(q.val);
        }
        return null;
    }
    public void preOrder(TreeNode node){
        if (node.left != null){
            parent.put(node.left.val, node);
            preOrder(node.left);
        }
        if (node.right != null){
            parent.put(node.right.val, node);
            preOrder(node.right);
        }
    }

    /**
     * 字典树
     */
    class TrieNode{
        private TrieNode[] childNodes;
        private final int num = 26;
        //标志位，用来解决两个单词前缀一样的问题：比如app和apple，不加标志位会导致只能查找到apple
        private boolean isEnd;

        public TrieNode(){
            childNodes = new TrieNode[num];
        }

        public boolean containKey(char ch){
            return childNodes[ch-'a'] != null;
        }

        public TrieNode get(char ch){
            return childNodes[ch-'a'];
        }

        public void insert(char ch,TrieNode node){
            childNodes[ch-'a']=node;
        }

        public void setEnd(){
            isEnd=true;
        }

        public boolean isEnd() {
            return isEnd;
        }
    }
    class Trie {

        private TrieNode root;
        /** Initialize your data structure here. */
        public Trie() {
            root = new TrieNode();
        }

        /** Inserts a word into the trie. */
        public void insert(String word) {
            TrieNode trieNode = root;
            for(int i = 0;i < word.length();i++){
                char ch = word.charAt(i);
                if(! trieNode.containKey(ch)){
                    trieNode.insert(ch,new TrieNode());
                }
                trieNode = trieNode.get(ch);
            }
            trieNode.setEnd();
        }

        /** Returns if the word is in the trie. */
        public boolean search(String word) {
            TrieNode trieNode = root;
            for(int i = 0;i < word.length();i++){
                char ch = word.charAt(i);
                if(! trieNode.containKey(ch)){
                    return false;
                }
                trieNode = trieNode.get(ch);
            }
            return trieNode != null && trieNode.isEnd();
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        public boolean startsWith(String prefix) {
            TrieNode trieNode = root;
            for(int i = 0;i < prefix.length();i++){
                char ch = prefix.charAt(i);
                if(! trieNode.containKey(ch)){
                    return false;
                }
                trieNode = trieNode.get(ch);
            }
            return trieNode != null;
        }
    }
    @Test
    public void test(){
        TreeNode i = null;
        TreeNode integer = new TreeNode(300);
        for (int j = 0; j < 2; j++){
            if (j == 1){
                i = integer;
            }
        }
        System.out.println(i);
    }
    public static void main(String[] args) {
        TreeNode t3 = new TreeNode(3);
        TreeNode t5 = new TreeNode(5);
        TreeNode t1 = new TreeNode(1);
        TreeNode t6 = new TreeNode(6);
        TreeNode t2 = new TreeNode(2);
        TreeNode t0 = new TreeNode(0);
        TreeNode t8 = new TreeNode(8);
        TreeNode t7 = new TreeNode(7);
        TreeNode t4 = new TreeNode(4);
        t3.left = t5;
        t3.right = t1;
        t5.left = t6;
        t5.right = t2;
        t1.left = t0;
        t1.right = t8;
        t2.left = t7;
        t2.right = t4;
        Code code = new Code();
        System.out.println(code.lowestCommonAncestor(t3, t5, t1).val);
//        buildTree2(new int[]{9,3,15,20,7}, new int[]{9,15,7,20,3});

//        ListNode lnf10 = new ListNode(-10);
//        ListNode lnf3 = new ListNode(-3);
//        ListNode ln0 = new ListNode(0);
//        ListNode ln5 = new ListNode(5);
//        ListNode ln9 = new ListNode(9);
//        lnf10.next = lnf3;
//        lnf3.next = ln0;
//        ln0.next = ln5;
//        ln5.next = ln9;
//        TreeNode node = new Code().sortedListToBST(lnf10);
//        System.out.println(node.val);
    }

}
