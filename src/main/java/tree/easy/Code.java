package tree.easy;

import org.omg.CORBA.MARSHAL;
import tree.TreeNode;

import javax.crypto.spec.DESedeKeySpec;
import javax.xml.soap.Node;
import java.util.*;
import java.util.concurrent.DelayQueue;

public class Code {
    //检查二叉树是否镜像对称
    public boolean isSymmetric(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)){
            return true;
        }
        if (root.left == null || root.right == null) {
            return false;
        }

        return check(root.left, root.right);
    }

    public boolean check(TreeNode l, TreeNode r){
        if (l == null && r == null){
            return true;
        }
        if (l == null || r == null){
            return false;
        }

        return l.val == r.val && check(l.left, r.right) && check(l.right, r.left);
    }

    //检验两棵树是否相同
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null){
            return true;
        }
        if (p == null || q == null){
            return false;
        }
        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //检验两棵树是否相同方法2
    public boolean isSameTree2(TreeNode p, TreeNode q) {
        if (p == null && q == null){
            return true;
        }
        if (p == null || q == null){
            return false;
        }
        Deque<TreeNode> nodeP = new LinkedList<>();
        Deque<TreeNode> nodeQ = new LinkedList<>();

        nodeP.offer(p);
        nodeQ.offer(q);
        while (!nodeP.isEmpty() && !nodeQ.isEmpty()){
            TreeNode removeP = nodeP.pollFirst();
            TreeNode removeQ = nodeQ.pollFirst();
            //如果值不相等返回false
            if (removeP.val != removeQ.val){
                return false;
            }else { //如果值相等则比较子节点为空的情况
                if (removeP.left == null && removeQ.left != null){
                    return false;
                }
                if (removeP.left != null && removeQ.left == null){
                    return false;
                }
                if (removeP.right == null && removeQ.right != null){
                    return false;
                }
                if (removeP.right != null && removeQ.right == null){
                    return false;
                }
            }
            if (removeP.left != null){
                nodeP.offerLast(removeP.left);
            }
            if (removeP.right != null){
                nodeP.offerLast(removeP.right);
            }
            if (removeQ.left != null){
                nodeQ.offerLast(removeQ.left);
            }
            if (removeQ.right != null){
                nodeQ.offerLast(removeQ.right);
            }
        }
        return nodeP.size() == nodeQ.size();
    }

    //二叉树最大深度
    int[] max = new int[]{0};
    int m = 0;
    public int maxDepth(TreeNode root) {
        if (root == null){
            return 0;
        }
        return maxDepth(root, max, m);
    }

    public int maxDepth(TreeNode node, int[] max, int m) {
        if (node == null){
            return 0;
        }
        m += 1;
        if (m > max[0]){
            max[0] = m;
        }
        return Math.max(maxDepth(node.left, max, m), maxDepth(node.right, max, m));
    }

    //给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0 || nums == null){
            return null;
        }
        int len = nums.length;
        return sortedArrayToBST(nums, 0, len - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int left, int right) {
        if (left > right){
            return null;
        }
        int mid = (left + right) / 2;
        int rootVal = nums[mid];
        TreeNode node = new TreeNode(rootVal);
        node.left = sortedArrayToBST(nums, left, mid - 1);
        node.right = sortedArrayToBST(nums, mid + 1, right);
        return node;
    }

    //给定一个二叉树，判断它是否是高度平衡的二叉树
    //一个二叉树 每个节点 的左右两个子树的高度差的绝对值不超过 1 。
    public boolean isBalanced(TreeNode root) {
        if (root == null){
            return true;
        }else {
            return Math.abs(height(root.left) - height(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
        }
    }
    public int height(TreeNode root) {
        if (root == null){
            return 0;
        }else {
            return Math.max(height(root.left), height(root.right)) + 1;
        }
    }

    //给定一个二叉树，找出其最小深度的叶子节点（递归）。
    public int minDepth(TreeNode root){
        if (root == null){
            return 0;
        }
        int left = minDepth(root.left);
        int right = minDepth(root.right);
        if (left == 0 || right == 0){
            return left + right + 1;
        }else {
            return Math.min(left, right) + 1;
        }
    }
    //方法2（广度优先搜索）
    class NodeDepth {
        public TreeNode node;
        public int depth;

        public NodeDepth(){
        }

        public NodeDepth(TreeNode node, int depth) {
            this.node = node;
            this.depth = depth;
        }
    }
    public int minDepth2(TreeNode root){
        if (root == null){
            return 0;
        }
        LinkedList<NodeDepth> queue = new LinkedList<>();
        NodeDepth rootDepth = new NodeDepth(root, 1);
        queue.offer(rootDepth);
        while (!queue.isEmpty()){
            NodeDepth nodeDepth = queue.pollFirst();
            TreeNode node = nodeDepth.node;
            int depth = nodeDepth.depth;
            if (node.left == null && node.right == null){
                return depth;
            }
            if (node.left != null){
                NodeDepth nodeDepthLeft = new NodeDepth(node.left, nodeDepth.depth + 1);
                queue.offerLast(nodeDepthLeft);
            }
            if (node.right != null){
                NodeDepth nodeDepthRight = new NodeDepth(node.right, nodeDepth.depth + 1);
                queue.offerLast(nodeDepthRight);
            }
        }
        return 0;
    }

    //路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null){
            return false;
        }else {
            if (root.left == null && root.right == null){
                if (root.val == targetSum){
                    return true;
                }else {
                    return false;
                }
            }else {
                return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
            }
        }
    }
    //方法2（广度优先搜索）
    public boolean hasPathSum2(TreeNode root, int targetSum) {
        if (root == null){
            return false;
        }
        Deque<TreeNode> queNode = new LinkedList<>();
        Deque<Integer> queVal = new LinkedList<>();

        queNode.offerLast(root);
        queVal.offerLast(root.val);

        while (!queNode.isEmpty()){
            TreeNode now = queNode.pollFirst();
            int tmp = queVal.pollFirst();
            if (now.left == null && now.right == null){
                if (tmp == targetSum){
                    return true;
                }
                continue;
            }
            if (now.left != null){
                queNode.offerLast(now.left);
                queVal.offerLast(now.left.val + tmp);
            }
            if (now.right != null){
                queNode.offerLast(now.right);
                queVal.offerLast(now.right.val + tmp);
            }
        }
        return false;
    }

    //二叉树前序遍历(递归)
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null){
            return list;
        }
        preorderTraversal(root, list);
        return list;
    }
    public void preorderTraversal(TreeNode root, List<Integer> list) {
        if (root == null){
            return;
        }
        list.add(root.val);
        preorderTraversal(root.left, list);
        preorderTraversal(root.right, list);
    }
    //方法2(迭代)
    public List<Integer> preorderTraversal2(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null){
            return list;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        TreeNode node = root;
        while (!deque.isEmpty() || node != null){
            while (node != null){
                deque.offerLast(root);
                list.add(root.val);
                root = root.left;
            }
            node = deque.pollLast();
            node = node.right;
        }
        return list;
    }

    /**
     * 二叉树后序遍历
     * @param root
     * @return
     */
    //方法1（递归）
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null){
            return list;
        }
        postorderTraversal(root, list);
        return list;
    }
    public void postorderTraversal(TreeNode node, List<Integer> list) {
        if (node == null){
            return;
        }
        preorderTraversal(node.left, list);
        preorderTraversal(node.right, list);
        list.add(node.val);
    }
    //方法2(迭代)
    public List<Integer> postorderTraversal2(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null){
            return list;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        TreeNode node = root;
        TreeNode pre = null;
        while (!deque.isEmpty() || node != null){
            while (node != null){
                deque.offerLast(node);
                node = node.left;
            }
            node = deque.pollLast();
            if (node.right == null || pre == node.right){
                list.add(node.val);
                pre = node;
                node = null;
            }else {
                deque.offerLast(node);
                node = node.right;
            }
        }
        return list;
    }
    //方法3(前序遍历（根、右、左）反转)
    public List<Integer> postorderTraversal3(TreeNode root) {
        LinkedList<Integer> list = new LinkedList<>();
        if (root == null){
            return list;
        }
        Deque<TreeNode> deque = new LinkedList<>();
        TreeNode node = root;
        while (!deque.isEmpty() || node != null){
            while (node != null){
                deque.offerLast(node);
                list.offerFirst(node.val);
                node = node.right;
            }

            node = deque.pollLast();
            node = node.left;

        }
        return list;
    }

    /**
     * 实现一个二叉搜索树迭代器类
     */
    class BSTIterator {

        TreeNode now;
        Deque<TreeNode> deque;
        public BSTIterator(TreeNode root) {
            this.now = root;
            this.deque = new LinkedList<>();
        }

        public int next() {
            while (now != null){
                deque.offerLast(now);
                now = now.left;
            }

            now = deque.pollLast();
            int res = now.val;
            now = now.right;
            return res;
        }

        public boolean hasNext() {
            return this.now != null || !this.deque.isEmpty();
        }
    }

    /**
     * 求出两节点的最近公共祖先（深度最大）
     * 思路：一次遍历，当pq值都小于当前节点值时，遍历左子树，pq值都大于当前节点值时，遍历右子树
     *      如果pq值一左一右，那么该节点值作为第一个分叉点即为最近公共祖先
     * @param root
     * @param p
     * @param q
     * @return
     */
    TreeNode res = null;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        lowestCommonAncestor2(root, p, q);
        return res;
    }
    public void lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {

        if (root == null){
            return;
        }

        if ((p.val - root.val) * (q.val - root.val) <= 0){
            res = root;
        }

        if (p.val > root.val && q.val > root.val){ //都在右子树
            lowestCommonAncestor2(root.right, p, q);
        }
        if (p.val < root.val && q.val < root.val){ //都在左子树
            lowestCommonAncestor2(root.left, p, q);
        }
    }

    /**
     * 返回所有从根节点到叶子节点的路径。
     * 思路1：前序遍历（递归）
     * 思路2：迭代
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> list = new ArrayList<>();
        binaryTreePaths2(root, "", list);
        return list;
    }

    public void binaryTreePaths2(TreeNode node, String path, List<String> list){
        if (node == null){
            return;
        }
//        StringBuilder sb = new StringBuilder(path);
//        sb.append(Integer.toString(node.val));
        path += Integer.toString(node.val);
        if (node.left == null && node.right == null){
            list.add(path);
        }else {
            path += "->";
            binaryTreePaths2(node.left, new String(path), list);
            binaryTreePaths2(node.right, new String(path), list);
        }
    }
    //方法2（迭代）
    public List<String> binaryTreePaths3(TreeNode root){
        List<String> list = new ArrayList<>();
        if (root == null){
            return list;
        }

        Deque<TreeNode> nodeQueue = new LinkedList<>();
        Deque<String> pathQueue = new LinkedList<>();

        nodeQueue.offerLast(root);
        pathQueue.offerLast(Integer.toString(root.val));
        TreeNode pollNode;
        String pollPath;
        while (!nodeQueue.isEmpty()){
            pollNode = nodeQueue.pollFirst();
            pollPath = pathQueue.pollFirst();
            if (pollNode.left == null && pollNode.right == null){
                list.add(pollPath);
            }
            if (pollNode.left != null){
                nodeQueue.offerLast(pollNode.left);
                pathQueue.offerLast(pollPath + "->" + pollNode.left.val);
            }
            if (pollNode.right != null){
                nodeQueue.offerLast(pollNode.right);
                pathQueue.offerLast(pollPath + "->" + pollNode.right.val);
            }
        }
        return list;
    }


    public static void main(String[] args) {
        TreeNode t1 = new TreeNode(1);
        TreeNode t2 = new TreeNode(2);
        TreeNode t3 = new TreeNode(2);
        TreeNode t4 = new TreeNode(3);
        TreeNode t5 = new TreeNode(3);
        TreeNode t6 = new TreeNode(4);
        TreeNode t7 = new TreeNode(4);
        t1.left = t2;
        t1.right = t3;
        t2.left = t4;
        t3.right = t5;
        t4.left = t6;
        t5.left = t7;
        boolean balanced = new Code().isBalanced(t1);
        System.out.println(balanced);
    }
}
